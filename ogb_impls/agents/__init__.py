from ogb_impls.agents.gciql import GCIQLAgent
from ogb_impls.agents.hiql1v import HIQL1vAgent
from ogb_impls.agents.hiql2v import HIQL2vAgent
from ogb_impls.agents.arl import ARLAgent

agents = dict(
    gciql=GCIQLAgent,
    hiql2v=HIQL2vAgent,
    hiql1v=HIQL1vAgent,
    arl=ARLAgent,
)