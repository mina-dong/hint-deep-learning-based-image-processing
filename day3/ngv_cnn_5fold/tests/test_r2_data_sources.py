"""R2-only regression tests. ALL image arrays are synthetic test fixtures.
Nothing generated here is part of the student data bundle or evidence of accuracy.
"""
import hashlib, io, json, tarfile, tempfile, unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from PIL import Image
from cnnlab import builder
from cnnlab.common import ROOT, read_json, sha_file, write_json
from cnnlab.data import DATASET_SIZES, CLASSES, save_bundle, load_bundle, make_split, folds_for, nested_subset


def fixture(n, shape, classes):
    rng=np.random.default_rng(982)
    return rng.integers(0,256,(n,*shape),dtype=np.uint8),np.arange(n,dtype=np.int64)%classes,np.asarray([f'SYNTHETIC_R2_{i}' for i in range(n)])


def tar_bytes(members):
    b=io.BytesIO()
    with tarfile.open(fileobj=b,mode='w:gz') as t:
        for name,payload in members:
            item=tarfile.TarInfo(name);item.size=len(payload);t.addfile(item,io.BytesIO(payload))
    return b.getvalue()


def oxford_fixture(root):
    members=[];lines=[];ids_to_species={}
    # Deliberately reverse the real dataset's capitalization convention: only annotation labels are authoritative.
    for c,base in [(0,'lowercase_cat'),(1,'Uppercase_Dog')]:
        for i in range(5):
            image_id=f'{base}_{i}';ids_to_species[image_id]=c
            x=np.random.default_rng(10*c+i).integers(0,256,(40,60,3),dtype=np.uint8)
            b=io.BytesIO();Image.fromarray(x).save(b,format='JPEG')
            members.append((f'images/{image_id}.jpg',b.getvalue()))
            lines.append(f'{image_id} {c+1} {c+1} {c+1}')
    # This image is NOT in trainval and must not be selected.
    members.append(('images/excluded_official_test_1.jpg',members[0][1]))
    image=root/'images.tar.gz';annotation=root/'annotations.tar.gz'
    image.write_bytes(tar_bytes(members))
    text=('\n'.join(lines)+'\n').encode()
    annotation.write_bytes(tar_bytes([('annotations/trainval.txt',text)]))
    return image,annotation,members,text,ids_to_species


class TestR2Counts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.x,cls.y,cls.ids=fixture(10000,(32,32,3),10)
    def test_01_target_counts(self):
        self.assertEqual(DATASET_SIZES,{'mnist':2000,'cifar10':10000,'catsdogs':2000})
    def test_02_cifar_fixed_80_20(self):
        s=make_split(self.y,self.ids)
        self.assertEqual((len(s['train_pool']),len(s['test'])),(8000,2000))
        self.assertFalse(set(s['train_pool'])&set(s['test']))
        self.assertTrue(np.all(np.bincount(self.y[s['test']])==200))
    def test_03_cifar_folds(self):
        s=make_split(self.y,self.ids);seen=[]
        pairs=folds_for(np.asarray(s['train_pool']),self.y,137)
        for a,b in pairs:
            self.assertEqual((len(a),len(b)),(6400,1600))
            self.assertFalse(set(a)&set(b));self.assertFalse((set(a)|set(b))&set(s['test']))
            seen.extend(b.tolist())
        self.assertEqual(len(set(seen)),8000)
    def test_04_cifar_loader_roundtrip_and_old_rejection(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);save_bundle(p,'cifar10',self.x,self.y,self.ids,{'kind':'SYNTHETIC_TEST_ONLY'})
            x,y,ids,m=load_bundle(p,'cifar10')
            self.assertEqual(m['n'],10000);self.assertIn('8000 development / 2000',m['protocol'])
            self.assertTrue(np.array_equal(x,self.x))
            with self.assertRaises(ValueError):load_bundle(p,'cifar10',expected=2000)
    def test_05_cifar_configs_change_only_count(self):
        for name in ['exp01_cifar10.json','exp03_models.json']:
            c=read_json(ROOT/'configs'/name)['base']
            self.assertEqual(c['train_count'],8000)
            self.assertEqual((c['epochs'],c['batch_size'],c['learning_rate']),(5,32,0.001))
    def test_06_subset_accepts_8000(self):
        s=make_split(self.y,self.ids)
        a=nested_subset(s['train_pool'],self.y,8000)
        self.assertEqual(len(a),8000)


class TestOxfordParser(unittest.TestCase):
    def test_07_species_from_annotations(self):
        with tempfile.TemporaryDirectory() as d:
            image,ann,_,_,labels=oxford_fixture(Path(d))
            x,y,ids=builder.read_oxford_pet(image,ann,per_class=3)
            self.assertEqual(x.shape,(6,96,96,3));self.assertEqual(np.bincount(y).tolist(),[3,3])
            for label,id_ in zip(y,ids):self.assertEqual(label,labels[id_.split(':',1)[1]])
            self.assertTrue(all('excluded_official_test' not in id_ for id_ in ids))
    def test_08_archive_directory_same_selection(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);image,ann,members,text,_=oxford_fixture(root)
            (root/'annotations').mkdir();(root/'annotations/trainval.txt').write_bytes(text)
            (root/'images').mkdir()
            for name,b in members:(root/name).write_bytes(b)
            a=builder.read_oxford_pet(image,ann,per_class=3)
            b=builder.read_oxford_pet(root/'images',root/'annotations',per_class=3)
            for x,y in zip(a,b):self.assertTrue(np.array_equal(x,y))
    def test_09_insufficient_unique_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            image,ann,*_=oxford_fixture(Path(d))
            with self.assertRaises(ValueError):builder.read_oxford_pet(image,ann,per_class=6)
    def test_10_bad_species_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/'trainval.txt').write_text('pet_1 1 3 1\n')
            with self.assertRaises(ValueError):builder.read_oxford_annotations(p)
    def test_11_duplicate_ids_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/'trainval.txt').write_text('pet_1 1 1 1\npet_1 1 1 1\n')
            with self.assertRaises(ValueError):builder.read_oxford_annotations(p)
    def test_12_source_missing_no_network(self):
        with tempfile.TemporaryDirectory() as d,patch('urllib.request.urlopen',side_effect=AssertionError('network forbidden')):
            with self.assertRaises(FileNotFoundError):builder.prepare_oxford_sources([Path(d)],Path(d),False)
    def test_13_official_archive_checksum_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'images.tar.gz';p.write_bytes(b'invalid')
            with self.assertRaisesRegex(ValueError,'MD5 mismatch'):builder._verify_oxford_archive(p,'images.tar.gz')
    def test_14_local_sources_with_mocked_fixture_checksums(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);image,ann,*_=oxford_fixture(p)
            resources={k:{**v,'md5':hashlib.md5((p/k).read_bytes()).hexdigest()} for k,v in builder.OXFORD_RESOURCES.items()}
            with patch.dict(builder.OXFORD_RESOURCES,resources),patch('urllib.request.urlopen',side_effect=AssertionError('network forbidden')):
                a,b=builder.prepare_oxford_sources([p],p,False)
                self.assertEqual((a,b),(image,ann))
    def test_15_download_fallback_mocked(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);image,ann,*_=oxford_fixture(p)
            payload={r:(p/r).read_bytes() for r in builder.OXFORD_RESOURCES}
            for r in payload:(p/r).unlink()
            def fake_download(url,dest,limit_mb):
                if 'thor.' in url:raise OSError('fixture primary host unavailable')
                dest.write_bytes(payload[dest.name]);return dest
            resources={k:{**v,'md5':hashlib.md5(payload[k]).hexdigest()} for k,v in builder.OXFORD_RESOURCES.items()}
            with patch.dict(builder.OXFORD_RESOURCES,resources),patch.object(builder,'download',side_effect=fake_download) as mock:
                a,b=builder.prepare_oxford_sources([p],p,True)
                self.assertEqual(mock.call_count,4)
                self.assertEqual((a,b),(image,ann))


class TestCifarPrefixAndMigration(unittest.TestCase):
    def test_16_prefix_across_batches_no_official_test(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);x,_,_=fixture(30,(32,32,3),10)
            labels=np.asarray([0]+list(range(1,10))*2+[0]+list(range(10)))
            def payload(a,b):
                return np.column_stack([labels[a:b].astype(np.uint8),x[a:b].transpose(0,3,1,2).reshape(b-a,-1)]).tobytes()
            raw=tar_bytes([('cifar-10-batches-bin/test_batch.bin',payload(0,10)),
                           ('cifar-10-batches-bin/data_batch_1.bin',payload(0,19)),
                           ('cifar-10-batches-bin/data_batch_2.bin',payload(19,30))])
            with patch('urllib.request.urlopen',return_value=io.BytesIO(raw)):
                a,b,ids=builder.cifar_prefix(p,per_class=2)
            self.assertEqual(len(b),20);self.assertTrue(np.all(np.bincount(b)==2))
            self.assertTrue(any('data_batch_2.bin' in i for i in ids))
            self.assertTrue(all('test_batch' not in i for i in ids))
            with patch('urllib.request.urlopen',side_effect=AssertionError('network forbidden')):
                aa,bb,ii=builder.cifar_prefix(p,per_class=2,allow_download=False)
            self.assertTrue(np.array_equal(a,aa));self.assertTrue(np.array_equal(b,bb));self.assertTrue(np.array_equal(ids,ii))
    def test_17_old_prefix_not_mistaken_for_new(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/'cifar_prefix_source.npz').write_bytes(b'old-small-cache')
            with patch('urllib.request.urlopen',side_effect=AssertionError('network forbidden')):
                with self.assertRaises(FileNotFoundError):builder.cifar_prefix(p,allow_download=False)
            self.assertEqual((p/'cifar_prefix_source.npz').read_bytes(),b'old-small-cache')
    def test_18_publish_failure_preserves_old_pair(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);data=p/'data';data.mkdir()
            (data/'cifar10.npz').write_bytes(b'oldnpz');(data/'cifar10.json').write_bytes(b'oldjson')
            with patch.object(builder,'save_bundle',side_effect=ValueError('fixture rejection')):
                with self.assertRaises(ValueError):builder._publish_bundle(data,'cifar10',None,None,None,{},p/'backup')
            self.assertEqual((data/'cifar10.npz').read_bytes(),b'oldnpz')
            self.assertEqual((data/'cifar10.json').read_bytes(),b'oldjson')
    def test_19_published_pair_backs_up_old(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);data=p/'data';data.mkdir()
            (data/'mnist.npz').write_bytes(b'oldnpz');(data/'mnist.json').write_bytes(b'oldjson')
            x,y,ids=fixture(2000,(28,28,1),10)
            builder._publish_bundle(data,'mnist',x,y,ids,{'kind':'SYNTHETIC_TEST_ONLY'},p/'backup')
            self.assertEqual(len(load_bundle(data,'mnist')[1]),2000)
            old=list((p/'backup').glob('*/mnist.npz'))
            self.assertEqual(len(old),1);self.assertEqual(old[0].read_bytes(),b'oldnpz')
    def test_20_protocol_text_dynamic(self):
        source=(ROOT/'cnnlab/runner.py').read_text()
        self.assertNotIn('internal_holdout400_after',source)
        report=(ROOT/'cnnlab/reporting.py').read_text()
        self.assertNotIn('시험 이미지 400장',report)

if __name__=='__main__':unittest.main()
