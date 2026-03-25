"""
Quick test to check what's in the Chl files
"""
import xarray as xr
from pathlib import Path

chl_file = list(Path("data/raw/chl").glob("*.nc"))[0]
print(f"Checking file: {chl_file.name}")

# Try different engines
engines = ['netcdf4', 'h5netcdf', 'scipy']

for engine in engines:
    try:
        ds = xr.open_dataset(chl_file, engine=engine)
        print(f"\n{engine} SUCCESS!")
        print(f"Variables: {list(ds.variables.keys())}")
        print(f"Data vars: {list(ds.data_vars.keys())}")
        print(f"Dims: {ds.dims}")
        ds.close()
        break
    except Exception as e:
        print(f"{engine}: {str(e)[:100]}")
