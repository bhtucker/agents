import pytest
import pickle

@pytest.fixture
def nnpop():
    # returns a broken nnpop (entities' task tracking has been frozen)    
    
    with open('/Users/bhtucker/rc/abm/agents/tests/abm/nnpop.pkl', 'rb') as f:
        pop = pickle.load(f)
    return pop
