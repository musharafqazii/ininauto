"""The covariant ACTION is the only required input to the pipeline."""
import sympy as sp
from ininauto import run,KIN,PHI2,RPHI2,PHI4

m, lam=sp.symbols('m lambda',real=True)
action=-KIN/2-m**2*PHI2/2-RPHI2/12-lam*PHI4/24

if __name__=='__main__':
    result=run(action)
    print('Output:',result.files)
    for name,data in result.families.items():
        print(name,'number of structures by order:',
              {k:len(v) for k,v in data[4].items()})
