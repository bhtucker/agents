# Agents
![Travis Badge](https://api.travis-ci.org/bhtucker/agents.svg?branch=master)

## Introduction

This is an agent-based model (ABM) in which the agents are embedded in a
network, one in each node. The main simulation consists of a package being
placed at a given starting node, `u`. The package has the identity of its
recipient, which `u` then reads and decides which of its adjacencies to hand the
package to, so that it may arrrive to its destination. Each agent only knows
about their immediate neighbors, not the global topology.

At first, the agents do not know how to use the information in the package to
correctly route it, but they are given learning mechanisms to improve their
decisions. This is done by using a reward system.

After a task is completed (package is delivered, or otherwise lost), the
environment gives a reward to each of the agents who participated in the
delivery chain. This is the only information each agent receives about the
success or failure of their decision.


## Purpose

Answer questions such as:

* How might different machine learning strategies lead to different agent behaviors?
* How do hardcoded learning mechanisms correlate to global routing efficiency?
* How do different network topology measures correlate with learning rates of individual agents?


## Details

#### Agents

Agents' identities are determined by a feature vector that defines their
characteristics. The package contains the feature vector of the recipient, which
the agent holding the package can read. From this information, the agent must
decide which of its adjacencies to hand the package to.

#### Networks

The project implements different ways to build networks with different
topologies, from the comletely random to drawing from edge probabilities based
on feature similarity of end nodes.


#### Learning

The project implements many different mechanisms in which agents decide who to
handle the message. Most importantly, by comparing feature vectors of the
recipient to those of its adjacencies. (Assuming that agents with similar
features are closer in the network.)


## Installing

If you have your numeric computing dependencies in order (ie libgfortran or gcc), simply `clone` the repo and `pip install .` from the project root.

This project uses `matplotlib`, so you may need to consult the [FAQ](matplotlib.org/faq/virtualenv_faq.html) if you're using virtualenv.

If you're using or would like to use `conda`, you may prefer to use the `conda install` command from `.travis.yml`, and then only use `pip` for `requirements-travis.txt`.


## Contributing

The modular implementation should make it possible for contributors to add new learning methods or network features.

The recommended contribution path is making a branch, trying out your feature in a script (see `/scripts`), and then discussing with an existing contributor or, if you feel comfortable, implementing it into `abm`. The next step would be an `analysis` script showing how this addition affects performance, adding `tests` to cover your additions, and a pull request.


## Example

```
>>> from abm import analysis, nxpops, io
>>> cfg = io.ConfigReader('../setup.json').get_config()
>>> pop = nxpops.SoftmaxNxEnvironment(**cfg)
>>> analysis.get_attrs(pop, 3)
{u'color': u'blue', u'region': u'east'}
```

## Generate documentation

Once you have forked this branch, you can do

```
$ sphinx-apidoc -o source/ abm -Ef
$ make html
```
