
```bash
module load modules/2.4-alpha2
module load python
python -m venv nanoenv
source nanoenv/bin/activate
```

Then the install package, note I'm also using a more recent and compatible torch:

```bash
pip install -r requirements.txt
pip install --pre torch==2.8.0.dev20250512+cu126 --index-url https://download.pytorch.org/whl/nightly/cu126 --upgrade
python data/cached_fineweb10B.py 20
```
