"""pick_instance.py: which IBM Quantum instance a live run uses. Shared by every demonstration.

DEMO_INSTANCE set: the instance whose name contains it (case ignored).
DEMO_INSTANCE not set: the key's only instance; if the key has several, the run stops and lists their names (never their CRNs).
24 Nov 2026: the earlier rule defaulted to an instance named "open-instance", which refused any key without one."""
import os, sys


def crn(base):
    insts = list(base.instances())
    want = os.environ.get('DEMO_INSTANCE', '').strip()
    if want:
        for i in insts:
            if want.lower() in str(i.get('name', '')).lower(): return i['crn']
        sys.exit("no instance named like '%s' on this key. Instances on this key: %s" % (want, ', '.join(str(i.get('name')) for i in insts)))
    if len(insts) == 1: return insts[0]['crn']
    if not insts: sys.exit('this key has no IBM Quantum instance')
    sys.exit('this key has %d instances: %s. Choose one:  export DEMO_INSTANCE="<name>"' % (len(insts), ', '.join(str(i.get('name')) for i in insts)))
