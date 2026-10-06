"""Teacher-only real-data preparation. Experiment scripts never access the network.
Network paths could not be integration-tested in the authoring environment.
Local NPZ / IDX / TAR / JPEG-ZIP parsers are covered by fixture tests.
"""
from __future__ import annotations
import gzip, hashlib, io, json, os, pickle, struct, tarfile, time, urllib.request, zipfile
from pathlib import Path, PurePosixPath
from datetime import datetime
import re, shutil, uuid
import numpy as np
from PIL import Image, ImageOps
from .common import ROOT, read_json, write_json, sha_file, project_lock
from .data import CLASSES, SHAPES, DATASET_SIZES, save_bundle, load_bundle, validate_arrays

MNIST_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
CIFAR_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz"
OXFORD_HOME = "https://www.robots.ox.ac.uk/~vgg/data/pets/"
OXFORD_DATASET = "Oxford-IIIT Pet"
# Official download-page host first; torchvision 0.25's official URLs second.
OXFORD_RESOURCES = {
    "images.tar.gz": {
        "urls": ["https://thor.robots.ox.ac.uk/datasets/pets/images.tar.gz",
                 "https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz"],
        "md5": "5c4f3ee8e5d25df40f4fd59a7f44e54c", "limit_mb": 850,
    },
    "annotations.tar.gz": {
        "urls": ["https://thor.robots.ox.ac.uk/datasets/pets/annotations.tar.gz",
                 "https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz"],
        "md5": "95a8c909bbe2e81eed6a22bccdf3f68f", "limit_mb": 50,
    },
}
CIFAR_PREFIX_CACHE = "cifar_prefix_source_10000.npz"
USER_AGENT = "NGV-CNN-Teaching/1.0"


def download(url: str, destination: Path, limit_mb=100):
    """Reuse completed files; resume a partial HTTP file only if a valid range is returned."""
    if destination.is_file():
        print(f"[LOCAL SOURCE] {destination}", flush=True)
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    part=destination.with_name(destination.name+".part")
    for attempt in range(2):
        start=part.stat().st_size if part.exists() else 0
        headers={"User-Agent":USER_AGENT}
        if start:
            headers["Range"]=f"bytes={start}-"
        print(f"[DOWNLOAD] {url} (attempt {attempt+1}/2)",flush=True)
        try:
            req=urllib.request.Request(url,headers=headers)
            with urllib.request.urlopen(req,timeout=30) as response:
                status=getattr(response,"status",200)
                append=(status==206 and response.headers.get("Content-Range","").startswith(f"bytes {start}-"))
                if status==206 and not append:
                    raise RuntimeError("Unexpected Content-Range. Remove the incomplete .part file and retry.")
                done=start if append else 0
                remaining=response.headers.get("Content-Length")
                total=done+int(remaining) if remaining else None
                if total and total>limit_mb*1024**2:
                    raise ValueError("Remote source exceeds the configured size limit")
                report_at=done
                with part.open("ab" if append else "wb") as f:
                    while True:
                        chunk=response.read(256*1024)
                        if not chunk:break
                        done+=len(chunk)
                        if done>limit_mb*1024**2:
                            raise ValueError("Download size limit exceeded")
                        f.write(chunk)
                        if done-report_at>=2*1024**2:
                            print(f"  {done/1024**2:.1f} MiB"+(f" / {total/1024**2:.1f} MiB" if total else ""),flush=True)
                            report_at=done
                if total is not None and done!=total:
                    raise IOError(f"Incomplete download: {done}/{total}")
            part.replace(destination)
            return destination
        except Exception as exc:
            if attempt==1:
                raise RuntimeError(f"Download failed: {url}\n{exc}\n"
                    f"You can instead place the original file at {destination}. No synthetic data is substituted.") from exc
            print(f"[RETRY] {exc}",flush=True)
            time.sleep(1)


def balanced_unique(images, labels, ids, domain, per_class, seed=20260929):
    rng=np.random.default_rng(seed)
    selected=[]
    seen=set()
    for c in range(len(CLASSES[domain])):
        count=0
        for i in rng.permutation(np.flatnonzero(np.asarray(labels)==c)):
            digest=hashlib.sha256(images[i].tobytes()).hexdigest()
            if digest in seen:continue
            seen.add(digest)
            selected.append(int(i));count+=1
            if count==per_class:break
        if count!=per_class:
            raise ValueError(f"{domain} class={c}: only {count} unique examples; require {per_class}. Do NOT duplicate samples.")
    selected=rng.permutation(selected)
    return np.asarray(images)[selected],np.asarray(labels)[selected],np.asarray(ids,dtype=str)[selected]


def read_mnist(path: Path):
    with np.load(path,allow_pickle=False) as z:
        x,y=z["x_train"],z["y_train"]
    if x.dtype!=np.uint8 or x.ndim!=3 or x.shape[1:]!=(28,28) or len(x)!=len(y):
        raise ValueError("Invalid source MNIST arrays")
    return x[:,:,:,None],y,np.asarray([f"official_train:{i}" for i in range(len(y))])


def read_idx(images_path, labels_path):
    def raw(p):
        return gzip.open(p,"rb") if str(p).endswith(".gz") else open(p,"rb")
    with raw(images_path) as f:
        magic,n,h,w=struct.unpack(">IIII",f.read(16))
        if (magic,h,w)!=(2051,28,28):raise ValueError("Invalid MNIST IDX image header")
        x=np.frombuffer(f.read(),dtype=np.uint8).reshape(n,28,28,1).copy()
    with raw(labels_path) as f:
        magic,n2=struct.unpack(">II",f.read(8))
        if magic!=2049 or n2!=n:raise ValueError("Invalid MNIST IDX label header")
        y=np.frombuffer(f.read(),dtype=np.uint8).copy()
    return x,y,np.asarray([f"official_train:{i}" for i in range(n)])


class RestrictedNumpyUnpickler(pickle.Unpickler):
    def find_class(self,module,name):
        if (module,name) in {("numpy.core.multiarray","_reconstruct"),("numpy._core.multiarray","_reconstruct"),
                             ("numpy","ndarray"),("numpy","dtype"),("_codecs","encode")}:
            return super().find_class(module,name)
        raise pickle.UnpicklingError(f"Blocked pickle object {module}.{name}")


def read_cifar_archive(path):
    xs,ys,ids=[],[],[]
    with tarfile.open(path,"r:gz") as tar:
        for m in tar:
            name=Path(m.name).name
            if not m.isfile() or not name.startswith("data_batch_"):
                continue
            f=tar.extractfile(m)  # No path extraction, so archive paths cannot escape the source directory.
            if name.endswith(".bin"):
                b=f.read()
                if len(b)%3073:raise ValueError("Incomplete CIFAR binary batch")
                arr=np.frombuffer(b,dtype=np.uint8).reshape(-1,3073)
                y=arr[:,0].copy();x=arr[:,1:].reshape(-1,3,32,32).transpose(0,2,3,1).copy()
            else:
                d=RestrictedNumpyUnpickler(f,encoding="latin1").load()
                y=np.asarray(d.get("labels",d.get(b"labels")))
                raw=d.get("data",d.get(b"data"))
                x=np.asarray(raw,dtype=np.uint8).reshape(-1,3,32,32).transpose(0,2,3,1).copy()
            xs.append(x);ys.append(y);ids.extend(f"{name}:{i}" for i in range(len(y)))
    if not xs:raise ValueError("No CIFAR training batches in archive")
    return np.concatenate(xs),np.concatenate(ys),np.asarray(ids)


class LimitedReader:
    def __init__(self, stream, limit=64*1024**2):
        self.stream,self.limit,self.count,self.next_report=stream,limit,0,2*1024**2
    def read(self,n=-1):
        if self.count>=self.limit:
            raise IOError(f"CIFAR prefix exceeded {self.limit/1024**2:g} MiB compressed transfer limit")
        n=min(n if n>=0 else 256*1024,self.limit-self.count)
        b=self.stream.read(n);self.count+=len(b)
        if self.count>=self.next_report:
            print(f"  CIFAR compressed bytes read: {self.count/1024**2:.1f} MiB",flush=True)
            self.next_report=self.count+2*1024**2
        return b


def cifar_prefix(source_dir, per_class=1000, allow_download=True):
    """Keep exactly 1000/class, stopping the stream when enough unique samples exist.
    A training batch is not exactly balanced: reading one 10000-image batch is
    not sufficient. Read small blocks across training batches; ignore test_batch.
    Network read-ahead/candidate images can exceed the retained 10000 samples.
    """
    source_dir=Path(source_dir)
    source_dir.mkdir(parents=True,exist_ok=True)
    cache=source_dir/CIFAR_PREFIX_CACHE
    if cache.is_file():
        with np.load(cache,allow_pickle=False) as z:
            cached=(z["images"],z["labels"],z["ids"])
        validate_arrays(*cached,"cifar10",expected=10*per_class)
        return cached
    if not allow_download:
        raise FileNotFoundError("No 10000-image CIFAR source cache. Supply an original CIFAR archive or enable ALLOW_DOWNLOAD. "
                                "The old 5000-record cifar_prefix_source.npz cannot supply 10000 distinct images.")
    print("[DOWNLOAD] CIFAR training prefix: retain 10000 images (1000/class); at most 64 MiB compressed bytes read.",flush=True)
    req=urllib.request.Request(CIFAR_URL,headers={"User-Agent":USER_AGENT})
    xs,ys,ids=[],[],[]
    try:
        with urllib.request.urlopen(req,timeout=30) as response:
            limited=LimitedReader(response)
            with tarfile.open(fileobj=limited,mode="r|gz") as tar:
                for m in tar:
                    name=PurePosixPath(m.name).name
                    if not m.isfile() or not re.fullmatch(r"data_batch_[1-5]\.bin",name):
                        continue
                    if m.size % 3073:
                        raise ValueError("Incomplete CIFAR binary batch")
                    offset=0
                    with tar.extractfile(m) as f:
                        while True:
                            block=f.read(500*3073)
                            if not block:break
                            if len(block)%3073:raise ValueError("Incomplete CIFAR prefix record")
                            arr=np.frombuffer(block,dtype=np.uint8).reshape(-1,3073)
                            y=arr[:,0].copy()
                            if np.any(y>9):raise ValueError("Invalid CIFAR class label")
                            x=arr[:,1:].reshape(-1,3,32,32).transpose(0,2,3,1).copy()
                            xs.append(x);ys.append(y)
                            ids.extend(f"{name}:{offset+i}" for i in range(len(y)))
                            offset+=len(y)
                            all_y=np.concatenate(ys)
                            if np.min(np.bincount(all_y,minlength=10)) < per_class:
                                continue
                            try:
                                selected=balanced_unique(np.concatenate(xs),all_y,np.asarray(ids),"cifar10",per_class)
                            except ValueError:
                                # Duplicate pixels can require a few extra candidate records.
                                continue
                            with cache.with_suffix(".tmp").open("wb") as out:
                                np.savez_compressed(out,images=selected[0],labels=selected[1],ids=selected[2])
                            cache.with_suffix(".tmp").replace(cache)
                            print(f"[PREFIX READY] kept={len(selected[1])}; candidate records read={len(all_y)}; "
                                  f"compressed bytes read={limited.count/1024**2:.1f} MiB",flush=True)
                            return selected
    except Exception as exc:
        raise RuntimeError(f"CIFAR prefix download failed: {exc}. Place a complete original CIFAR binary or Python archive "
                           "in teacher_sources and rerun. No full-download fallback starts automatically.") from exc
    raise ValueError("Not enough unique images per class in CIFAR training stream")


def read_cats_zip(path, per_class=1000, seed=20260929):
    """Legacy local parser only; the R2 builder never calls this or downloads its old source."""
    rng=np.random.default_rng(seed)
    xs,ys,ids=[],[],[]
    seen=set()
    with zipfile.ZipFile(path) as z:
        for c,word in enumerate(["cat","dog"]):
            names=sorted(n for n in z.namelist() if Path(n).name.startswith(word+".") and n.lower().endswith((".jpg",".jpeg",".png")))
            rng.shuffle(names)
            count=0
            for n in names:
                if z.getinfo(n).file_size>20*1024**2:continue
                try:
                    with z.open(n) as f:
                        with Image.open(f) as im:
                            # Fixed preprocessing performed before split; contains no label-derived statistics.
                            a=np.asarray(ImageOps.fit(im.convert("RGB"),(96,96),method=Image.Resampling.BILINEAR),dtype=np.uint8)
                except (OSError,ValueError):
                    continue
                digest=hashlib.sha256(a.tobytes()).hexdigest()
                if digest in seen:continue
                seen.add(digest);xs.append(a.copy());ys.append(c);ids.append(n);count+=1
                if count==per_class:break
            if count!=per_class:raise ValueError(f"Only {count} readable unique {word} images, need {per_class}")
    order=rng.permutation(len(ys))
    return np.asarray(xs)[order],np.asarray(ys)[order],np.asarray(ids)[order]



def read_oxford_annotations(path):
    """Species comes from column 3 of annotations/trainval.txt: 1=cat, 2=dog.
    Use the official trainval partition only; this lab makes its own 80/20 split.
    No breed-name capitalization heuristic and no 37-breed target substitution.
    """
    path=Path(path)
    if path.is_dir():
        candidate=path/"trainval.txt"
        if not candidate.is_file():candidate=path/"annotations"/"trainval.txt"
        text=candidate.read_text(encoding="utf-8")
    else:
        text=None
        with tarfile.open(path,"r:gz") as tar:
            for m in tar:
                if m.isfile() and PurePosixPath(m.name).as_posix().lstrip("./")=="annotations/trainval.txt":
                    if m.size>2*1024**2:raise ValueError("Unexpectedly large Oxford annotation list")
                    with tar.extractfile(m) as f:text=f.read().decode("utf-8")
                    break
        if text is None:raise ValueError("Oxford archive has no annotations/trainval.txt")
    labels={}
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):continue
        parts=line.split()
        if len(parts)!=4 or not re.fullmatch(r"[A-Za-z0-9_]+",parts[0]):
            raise ValueError(f"Invalid Oxford annotation row: {line!r}")
        image_id,breed,species,breed_within_species=parts
        if int(species) not in (1,2):raise ValueError("Oxford species label must be 1 (cat) or 2 (dog)")
        if image_id in labels:raise ValueError("Duplicate Oxford annotation image ID")
        labels[image_id]=int(species)-1
    if not labels:raise ValueError("Oxford trainval annotation list is empty")
    return labels


def read_oxford_pet(images_path, annotations_path, per_class=500, seed=20260929):
    """Read once without tar extraction. Local archives or extracted folders work.
    Preserve the R1 preprocessing: RGB + central fit to 96x96, bilinear.
    """
    labels=read_oxford_annotations(annotations_path)
    xs,ys,ids=[],[],[]
    def add(image_id, handle):
        try:
            with Image.open(handle) as im:
                a=np.asarray(ImageOps.fit(im.convert("RGB"),(96,96),method=Image.Resampling.BILINEAR),dtype=np.uint8)
        except (OSError,ValueError):
            print(f"[SKIP unreadable Oxford image] {image_id}",flush=True)
            return
        xs.append(a.copy());ys.append(labels[image_id]);ids.append(f"oxford_trainval:{image_id}")
    images_path=Path(images_path)
    if images_path.is_dir():
        image_dir=images_path/"images" if (images_path/"images").is_dir() else images_path
        for image_id in sorted(labels):
            p=image_dir/f"{image_id}.jpg"
            if p.is_file() and p.stat().st_size<=20*1024**2:
                with p.open("rb") as f:add(image_id,f)
    else:
        with tarfile.open(images_path,"r|gz") as tar:
            seen_names=set()
            for m in tar:
                parts=PurePosixPath(m.name).parts
                if not m.isfile() or m.size>20*1024**2 or len(parts)!=2 or parts[0]!="images":continue
                p=PurePosixPath(m.name)
                if p.suffix.lower()!=".jpg" or p.stem not in labels:continue
                if p.stem in seen_names:raise ValueError("Duplicate Oxford image archive member")
                seen_names.add(p.stem)
                with tar.extractfile(m) as f:add(p.stem,f)
    if not xs:raise ValueError("No readable Oxford images matching trainval annotations")
    # Identical selection for archive and extracted-folder input.
    order=np.argsort(np.asarray(ids))
    return balanced_unique(np.asarray(xs)[order],np.asarray(ys)[order],np.asarray(ids)[order],
                           "catsdogs",per_class,seed)


def _verify_oxford_archive(path, resource):
    if Path(path).is_dir():return
    expected=OXFORD_RESOURCES[resource]["md5"]
    with Path(path).open("rb") as f:
        actual=hashlib.file_digest(f,"md5").hexdigest()
    if actual!=expected:
        raise ValueError(f"Oxford {resource} MD5 mismatch: {path}. Keep only a complete official archive, "
                         "or supply verified extracted images/annotations folders.")


def prepare_oxford_sources(dirs, sources, allow_download):
    """Download the two Oxford official archives once. Approximately 800 MB total,
    NOT a 2000-image-sized network download. Existing completed sources are reused.
    """
    resolved={}
    for resource in ("annotations.tar.gz","images.tar.gz"):
        folder=resource.split(".")[0]
        candidates=[d/n for d in dirs for n in (resource,f"oxford-iiit-pet/{resource}",folder,f"oxford-iiit-pet/{folder}")]
        found=next((p for p in candidates if p.is_file() or p.is_dir()),None)
        if found is None:
            if not allow_download:
                raise FileNotFoundError(f"Oxford source missing: {resource}. Place images.tar.gz and annotations.tar.gz "
                                        "in teacher_sources (or set SOURCE_DIR), or enable ALLOW_DOWNLOAD.")
            errors=[]
            info=OXFORD_RESOURCES[resource]
            for url in info["urls"]:
                try:
                    found=download(url,sources/resource,limit_mb=info["limit_mb"])
                    _verify_oxford_archive(found,resource)
                    break
                except Exception as exc:
                    errors.append(str(exc));found=None
                    if (sources/resource).is_file():
                        # Do not silently accept or delete a completed but mismatched file.
                        raise
            if found is None:raise RuntimeError("Oxford download failed. Use local official archives.\n"+"\n".join(errors))
        _verify_oxford_archive(found,resource)
        resolved[folder]=found
    return resolved["images"],resolved["annotations"]


def _publish_bundle(data,domain,x,y,ids,source,backup_root):
    """Verify new files before replacing old pairs; keep previous data outside
    the student package. Unchanged MNIST is never regenerated or moved.
    """
    staging=backup_root/(".staging_"+domain+"_"+uuid.uuid4().hex[:8])
    staging.mkdir(parents=True,exist_ok=False)
    try:
        save_bundle(staging,domain,x,y,ids,source)
        load_bundle(staging,domain)
        old=[data/f"{domain}.{ext}" for ext in ("npz","json") if (data/f"{domain}.{ext}").exists()]
        if old:
            backup=backup_root/(datetime.now().strftime("%Y%m%d_%H%M%S")+"_"+domain+"_"+uuid.uuid4().hex[:6])
            backup.mkdir(parents=True,exist_ok=False)
            for p in old:shutil.move(str(p),str(backup/p.name))
            print(f"[BACKUP] Previous {domain} pair: {backup}",flush=True)
        data.mkdir(parents=True,exist_ok=True)
        for ext in ("npz","json"):
            (staging/f"{domain}.{ext}").replace(data/f"{domain}.{ext}")
    finally:
        shutil.rmtree(staging,ignore_errors=True)


def find_source(dirs,names):
    for d in dirs:
        for n in names:
            p=d/n
            if p.is_file():return p
    return None


def build_all(root=ROOT, source_dir=None, allow_download=False):
    root=Path(root)
    sources=Path(source_dir).expanduser() if source_dir else root/"teacher_sources"
    sources.mkdir(parents=True,exist_ok=True)
    dirs=[sources,root/"teacher_sources"]
    data=root/"data"
    with project_lock(root/"results"):
        for domain in ["mnist","cifar10","catsdogs"]:
            if (data/f"{domain}.npz").exists():
                old_meta=read_json(data/f"{domain}.json")
                old_n=int(old_meta["n"])
                # Validate the old pair before deciding whether it needs migration.
                load_bundle(data,domain,expected=old_n)
                current=(old_n==DATASET_SIZES[domain] and
                         (domain!="catsdogs" or old_meta.get("source",{}).get("dataset")==OXFORD_DATASET))
                if current:
                    print(f"[ALREADY VERIFIED] {domain}",flush=True)
                    continue
                if domain=="mnist":raise ValueError("Unexpected MNIST count; restore the R1 2000-image pair.")
                print(f"[UPDATE REQUIRED] {domain}: rebuilding requested count/source. Old files are retained until the new pair passes checks.",flush=True)
            print(f"\n[PREPARE] {domain}: exactly {DATASET_SIZES[domain]} distinct real examples",flush=True)
            if domain=="mnist":
                p=find_source(dirs,["mnist.npz"])
                images_path=find_source(dirs,["MNIST/raw/train-images-idx3-ubyte","MNIST/raw/train-images-idx3-ubyte.gz",
                                             "train-images-idx3-ubyte","train-images-idx3-ubyte.gz"])
                labels_path=find_source(dirs,["MNIST/raw/train-labels-idx1-ubyte","MNIST/raw/train-labels-idx1-ubyte.gz",
                                             "train-labels-idx1-ubyte","train-labels-idx1-ubyte.gz"])
                if p:
                    x,y,ids=read_mnist(p)
                elif images_path and labels_path:
                    x,y,ids=read_idx(images_path,labels_path);p=images_path
                elif allow_download:
                    p=download(MNIST_URL,sources/"mnist.npz",limit_mb=20);x,y,ids=read_mnist(p)
                else:
                    raise FileNotFoundError("MNIST source missing. Add teacher_sources/mnist.npz, or explicitly enable ALLOW_DOWNLOAD.")
                x,y,ids=balanced_unique(x,y,ids,domain,200)
                source={"url":MNIST_URL,"source_file":p.name,"source_sha256":sha_file(p),"sampling":"balanced random unique subset of official train"}
            elif domain=="cifar10":
                p=find_source(dirs,["cifar-10-binary.tar.gz","cifar-10-python.tar.gz"])
                if p:
                    x,y,ids=read_cifar_archive(p)
                    source={"url":"https://www.cs.toronto.edu/~kriz/cifar.html","source_file":p.name,
                            "source_sha256":sha_file(p),"sampling":"balanced random unique subset of available official training batches"}
                elif (sources/CIFAR_PREFIX_CACHE).is_file() or allow_download:
                    x,y,ids=cifar_prefix(sources,allow_download=allow_download)
                    source={"url":CIFAR_URL,"source_file":CIFAR_PREFIX_CACHE,
                        "source_sha256":sha_file(sources/CIFAR_PREFIX_CACHE),
                        "sampling":"1000 unique images/class selected from a prefix across official training batches; stopped after enough candidates; NOT random across the entire official training corpus"}
                else:
                    raise FileNotFoundError("CIFAR source missing. Add original CIFAR archive or explicitly enable ALLOW_DOWNLOAD.")
                x,y,ids=balanced_unique(x,y,ids,domain,1000)
            else:
                image_source,annotation_source=prepare_oxford_sources(dirs,sources,allow_download)
                x,y,ids=read_oxford_pet(image_source,annotation_source)
                source={"dataset":OXFORD_DATASET,"url":OXFORD_HOME,
                    "source_file":str(image_source),"annotation_source":str(annotation_source),
                    "source_sha256":sha_file(image_source) if image_source.is_file() else None,
                    "annotation_sha256":sha_file(annotation_source) if annotation_source.is_file() else None,
                    "official_split":"trainval", "label_mapping":{"1":"cat (0)","2":"dog (1)"},
                    "sampling":"1000 unique images/species from Oxford trainval; a new fixed 80/20 internal split is created",
                    "preprocessing":"RGB, central aspect-preserving fit to 96x96 using PIL bilinear",
                    "license":"CC BY-SA 4.0", "license_url":"https://creativecommons.org/licenses/by-sa/4.0/",
                    "attribution":"Oxford-IIIT Pet: O. M. Parkhi, A. Vedaldi, A. Zisserman, C. V. Jawahar, Cats and Dogs, CVPR 2012. Copyright remains with original image owners.",
                    "changes":"2000-image subset, species labels, RGB 96x96 center-fit/bilinear, NPZ repackaging, new internal split"}
            _publish_bundle(data,domain,x,y,ids,source,root/"teacher_sources"/"bundle_backups")
            _,_,_,meta=load_bundle(data,domain)
            print(f"[OK] {domain}: images={len(y)}, classes={np.bincount(y).tolist()}, "
                  f"train pool={len(meta['split']['train_pool'])}, fixed test={len(meta['split']['test'])}",flush=True)
        (data/"README.txt").write_text(
            "Verified bundles: MNIST 2000, CIFAR-10 10000, Oxford-IIIT Pet cat/dog 2000. See each JSON for source, checksums and fixed split.\n"
            "Oxford-IIIT Pet attribution: O. M. Parkhi, A. Vedaldi, A. Zisserman, C. V. Jawahar. Cats and Dogs, CVPR 2012.\n"
            "Source: https://www.robots.ox.ac.uk/~vgg/data/pets/ ; license: CC BY-SA 4.0, https://creativecommons.org/licenses/by-sa/4.0/ .\n"
            "Pet-image changes: 2000-image subset, species labels, RGB 96x96 bilinear center-fit, NPZ packaging, new 80/20 split.\n"
            "Copyright remains with original image owners. Keep this notice and each JSON with redistributed pet images.\n",encoding="utf-8")
        write_json(root/"BUNDLE_STATUS.json",{"data_included":True,"datasets":list(CLASSES),"images_per_dataset":DATASET_SIZES,
                    "network_required_for_training":False})
    print("\nAll real datasets verified. Run 08_make_student_zip.py to create a data-included classroom ZIP.",flush=True)


def make_student_zip(root=ROOT):
    root=Path(root)
    for domain in CLASSES:
        _,_,_,meta=load_bundle(root/"data",domain)
        if domain=="catsdogs" and meta.get("source",{}).get("dataset")!=OXFORD_DATASET:
            raise ValueError("R2 student ZIP requires Oxford-IIIT Pet catsdogs. Run 00_build_data_bundle.py to update the old source.")
    status={"data_included":True,"datasets":list(CLASSES),"images_per_dataset":DATASET_SIZES,"network_required_for_training":False}
    write_json(root/"BUNDLE_STATUS.json",status)
    out=root/"distribution"
    out.mkdir(exist_ok=True)
    path=out/"NGV_CNN_5Fold_WITH_DATA.zip"
    excludes={".venv","teacher_sources","results","distribution","__pycache__",".git",".pytest_cache"}
    temp=path.with_suffix(".tmp")
    with zipfile.ZipFile(temp,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(root.rglob("*")):
            rel=p.relative_to(root)
            if p.is_file() and not (set(rel.parts)&excludes) and p.suffix not in {".pyc",".part"}:
                z.write(p,Path("ngv_cnn_5fold")/rel)
    temp.replace(path)
    write_json(out/"package_integrity.json",{"filename":path.name,"sha256":sha_file(path),"bytes":path.stat().st_size})
    print(f"[WITH_DATA ZIP] {path}\nSize: {path.stat().st_size/1024**2:.1f} MiB",flush=True)
    return path
