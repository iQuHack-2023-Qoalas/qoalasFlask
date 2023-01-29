import covalent as ct
import time

@ct.electron
def wait(x):
    time.sleep(x)

@ct.lattice
def w(x):
    return wait(x)


id=ct.dispatch(w)(4)

print("here")
print(ct.get_result(id,wait=True))