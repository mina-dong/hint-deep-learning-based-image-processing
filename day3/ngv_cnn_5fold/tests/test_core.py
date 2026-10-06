"""All fixtures here are SYNTHETIC for code tests. Not dataset samples or performance evidence."""
import copy, io, json, pickle, struct, tarfile, tempfile, unittest, zipfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from PIL import Image
from cnnlab.common import ROOT, ids_hash, project_lock, read_json, write_json
from cnnlab.data import (make_split,nested_subset,folds_for,validate_arrays,save_bundle,load_bundle,make_loader)
from cnnlab.models import build_model, NAMES
from cnnlab.engine import evaluate
from cnnlab.runner import validate_config
from cnnlab.builder import (balanced_unique,read_idx,read_cifar_archive,read_cats_zip,read_mnist,
                             RestrictedNumpyUnpickler, LimitedReader, build_all, make_student_zip)


def fixture(n=2000,shape=(28,28,1),classes=10):
    rng=np.random.default_rng(2026)
    return (rng.integers(0,256,(n,*shape),dtype=np.uint8),np.arange(n,dtype=np.int64)%classes,
            np.asarray([f'SYNTHETIC_FIXTURE_{i}' for i in range(n)]))


class TestSplits(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.x,cls.y,cls.ids=fixture()
        cls.split=make_split(cls.y,cls.ids)
        cls.pool=np.array(cls.split['train_pool'])
        cls.test=np.array(cls.split['test'])
    def test_01_counts(self):
        self.assertEqual(len(self.pool),1600);self.assertEqual(len(self.test),400)
    def test_02_disjoint(self):
        self.assertFalse(set(self.pool)&set(self.test))
        self.assertEqual(len(set(self.pool)|set(self.test)),2000)
    def test_03_balance(self):
        self.assertTrue(np.all(np.bincount(self.y[self.test])==40))
    def test_04_deterministic(self):
        self.assertEqual(self.split,make_split(self.y,self.ids))
    def test_05_nested(self):
        sets=[set(nested_subset(self.pool,self.y,n)) for n in [200,400,800,1600]]
        self.assertTrue(all(a<=b for a,b in zip(sets,sets[1:])))
    def test_06_nested_balanced(self):
        s=nested_subset(self.pool,self.y,200)
        self.assertTrue(np.all(np.bincount(self.y[s])==20))
    def test_07_fold_count(self):
        pairs=folds_for(self.pool,self.y,137)
        self.assertEqual(len(pairs),5)
        self.assertTrue(all(len(a)==1280 and len(b)==320 for a,b in pairs))
    def test_08_oof_once(self):
        val=np.concatenate([b for a,b in folds_for(self.pool,self.y,137)])
        self.assertEqual(sorted(val),sorted(self.pool))
        self.assertEqual(len(set(val)),1600)
    def test_09_no_leakage(self):
        for a,b in folds_for(self.pool,self.y,137):
            self.assertFalse(set(a)&set(b));self.assertFalse(set(self.test)&(set(a)|set(b)))
    def test_10_same_test_across_sizes(self):
        before=ids_hash(self.ids[self.test])
        for n in [200,400,800,1600]:
            nested_subset(self.pool,self.y,n)
            self.assertEqual(before,ids_hash(self.ids[self.test]))
    def test_11_bad_size(self):
        for n in [0,13,199,2001]:
            with self.assertRaises(ValueError):nested_subset(self.pool,self.y,n)


class TestData(unittest.TestCase):
    def setUp(self):
        self.x,self.y,self.ids=fixture()
    def test_12_shape(self):
        validate_arrays(self.x,self.y,self.ids,'mnist')
        with self.assertRaises(ValueError):validate_arrays(self.x[:20],self.y[:20],self.ids[:20],'mnist')
    def test_13_duplicates(self):
        self.x[1]=self.x[0]
        with self.assertRaises(ValueError):validate_arrays(self.x,self.y,self.ids,'mnist')
    def test_14_id_collision(self):
        self.ids[1]=self.ids[0]
        with self.assertRaises(ValueError):validate_arrays(self.x,self.y,self.ids,'mnist')
    def test_15_bundle_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);save_bundle(p,'mnist',self.x,self.y,self.ids,{'kind':'SYNTHETIC_UNIT_TEST_ONLY'})
            x,y,ids,m=load_bundle(p,'mnist')
            self.assertTrue(np.array_equal(self.x,x));self.assertEqual(len(m['split']['test']),400)
    def test_16_checksum(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);save_bundle(p,'mnist',self.x,self.y,self.ids,{'kind':'SYNTHETIC_UNIT_TEST_ONLY'})
            with (p/'mnist.npz').open('ab') as f:f.write(b'corrupt')
            with self.assertRaises(ValueError):load_bundle(p,'mnist')
    def test_17_modified_split(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);save_bundle(p,'mnist',self.x,self.y,self.ids,{'kind':'SYNTHETIC_UNIT_TEST_ONLY'})
            m=read_json(p/'mnist.json');m['split']['seed']+=1;write_json(p/'mnist.json',m)
            with self.assertRaises(ValueError):load_bundle(p,'mnist')
    def test_18_missing_no_download(self):
        with tempfile.TemporaryDirectory() as d, patch('urllib.request.urlopen',side_effect=AssertionError('network not allowed')):
            with self.assertRaises(FileNotFoundError):load_bundle(Path(d),'mnist')
            with self.assertRaises(FileNotFoundError):build_all(Path(d),allow_download=False)
    def test_19_student_zip_blocks_missing_data(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(FileNotFoundError):make_student_zip(Path(d))
    def test_20_selection_unique_and_real_indices(self):
        a,b,ids=balanced_unique(self.x,self.y,self.ids,'mnist',100)
        self.assertEqual(len(b),1000);self.assertTrue(set(ids)<=set(self.ids))
        self.assertTrue(np.all(np.bincount(b)==100))
    def test_21_insufficient_samples(self):
        with self.assertRaises(ValueError):balanced_unique(self.x,self.y,self.ids,'mnist',201)
    def test_22_loader_shape(self):
        dl=make_loader(self.x,self.y,np.arange(33),'mnist',16,False,42)
        rows=list(dl)
        self.assertEqual(rows[0][0].shape,(16,1,28,28));self.assertEqual(rows[-1][0].shape[0],1)


class TestSources(unittest.TestCase):
    def test_23_mnist_npz(self):
        x,y,ids=fixture(100)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'mnist.npz';np.savez(p,x_train=x[:,:,:,0],y_train=y)
            a,b,c=read_mnist(p);self.assertTrue(np.array_equal(a,x))
    def test_24_mnist_idx(self):
        x,y,ids=fixture(100)
        with tempfile.TemporaryDirectory() as d:
            a,b=Path(d)/'images',Path(d)/'labels'
            a.write_bytes(struct.pack('>IIII',2051,100,28,28)+x.tobytes())
            b.write_bytes(struct.pack('>II',2049,100)+y.astype(np.uint8).tobytes())
            ax,ay,_=read_idx(a,b);self.assertTrue(np.array_equal(ax,x));self.assertTrue(np.array_equal(ay,y))
    def test_25_cifar_binary_tar(self):
        x,y,_=fixture(100,(32,32,3))
        b=np.column_stack([y.astype(np.uint8),x.transpose(0,3,1,2).reshape(100,-1)]).tobytes()
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'cifar.tar.gz'
            with tarfile.open(p,'w:gz') as tar:
                t=tarfile.TarInfo('cifar-10-batches-bin/data_batch_1.bin');t.size=len(b);tar.addfile(t,io.BytesIO(b))
            ax,ay,_=read_cifar_archive(p);self.assertTrue(np.array_equal(ax,x));self.assertTrue(np.array_equal(ay,y))
    def test_26_cifar_python_tar(self):
        x,y,_=fixture(100,(32,32,3))
        b=pickle.dumps({'data':x.transpose(0,3,1,2).reshape(100,-1),'labels':y.tolist()},protocol=2)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'cifar.tar.gz'
            with tarfile.open(p,'w:gz') as tar:
                t=tarfile.TarInfo('cifar-10-batches-py/data_batch_1');t.size=len(b);tar.addfile(t,io.BytesIO(b))
            ax,ay,_=read_cifar_archive(p);self.assertTrue(np.array_equal(ax,x))
    def test_27_pickle_blocked(self):
        import datetime
        b=pickle.dumps(datetime.datetime.now())
        with self.assertRaises(pickle.UnpicklingError):RestrictedNumpyUnpickler(io.BytesIO(b)).load()
    def test_28_cat_zip_parser(self):
        x,y,_=fixture(24,(80,120,3),2)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'pets.zip'
            with zipfile.ZipFile(p,'w') as z:
                for i,a in enumerate(x):
                    b=io.BytesIO();Image.fromarray(a).save(b,format='JPEG')
                    word='cat' if y[i]==0 else 'dog'
                    z.writestr(f'cats_and_dogs_filtered/train/{word}s/{word}.{i}.jpg',b.getvalue())
            ax,ay,ids=read_cats_zip(p,per_class=10)
            self.assertEqual(ax.shape,(20,96,96,3));self.assertTrue(np.all(np.bincount(ay)==10))
    def test_29_limited_reader(self):
        r=LimitedReader(io.BytesIO(b'a'*20),limit=10)
        self.assertEqual(len(r.read(30)),10)
        with self.assertRaises(IOError):r.read(1)


class TestModelsAndControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)
    def test_30_all_models_forward(self):
        for name in NAMES:
            m=build_model(name,3,10).eval()
            with torch.no_grad():z=m(torch.rand(2,3,32,32))
            self.assertEqual(z.shape,(2,10),name);self.assertTrue(torch.isfinite(z).all())
    def test_31_training_backward(self):
        for name in ['lenet_small','tiny_vgg','tiny_mobile','tiny_densenet']:
            m=build_model(name,3,2).train();z=m(torch.rand(2,3,32,32));torch.nn.functional.cross_entropy(z,torch.tensor([0,1])).backward()
            self.assertTrue(any(p.grad is not None for p in m.parameters()))
    def test_32_fresh_models_not_same_objects(self):
        a,b=build_model('lenet_small',1,10),build_model('lenet_small',1,10)
        self.assertIsNot(next(a.parameters()),next(b.parameters()))
    def test_33_metric_not_forced(self):
        class AlwaysOne(torch.nn.Module):
            def forward(self,x):return torch.tensor([[0.,1.]],device=x.device).repeat(len(x),1)
        dl=torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.ones(4,1,2,2),torch.tensor([0,1,0,1])),batch_size=2)
        m,p,y=evaluate(AlwaysOne(),dl,torch.device('cpu'))
        self.assertEqual(m['accuracy'],.5)
    def test_34_config_errors(self):
        cfg=read_json(ROOT/'configs/exp01_mnist.json')['base'];cfg['name']='test';validate_config(cfg)
        with self.assertRaises(ValueError):validate_config({**cfg,'folds':4})
        with self.assertRaises(ValueError):validate_config({**cfg,'dropout':1.})
    def test_35_lock(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)
            with project_lock(p):
                with self.assertRaises(RuntimeError):
                    with project_lock(p):pass
            self.assertFalse((p/'.running.lock').exists())

if __name__=='__main__':unittest.main()
