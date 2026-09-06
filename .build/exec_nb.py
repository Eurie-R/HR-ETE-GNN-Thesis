import sys, nbformat
from nbclient import NotebookClient
p = sys.argv[1]; to = int(sys.argv[2]) if len(sys.argv)>2 else 900
nb = nbformat.read(p, as_version=4)
# strip the pip cell for local exec (deps already present)
for c in nb.cells:
    if c.cell_type=='code' and c.source.strip().startswith('%pip'):
        c.source = "# (pip cell skipped locally)"
cl = NotebookClient(nb, timeout=to, kernel_name='python3', allow_errors=True)
cl.execute()
nbformat.write(nb, p)
bad=0
for i,c in enumerate(nb.cells):
    if c.cell_type!='code': continue
    for o in c.get('outputs',[]):
        if o.output_type=='error':
            bad+=1
            print(f"--- CELL {i} ERROR: {o.ename}: {o.evalue}")
            print('\n'.join(o.traceback[-6:]))
print(f"\n{'FAILED: '+str(bad)+' cells errored' if bad else 'ALL CELLS OK'}")
sys.exit(1 if bad else 0)
