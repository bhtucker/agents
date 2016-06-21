# -*- coding: utf-8 -*-
"""
    abm.analysis
    ~~~~~~~~~~~~

    Some functions for training/analyzing networks
"""
import operator
from abm import learners
import random
import networkx as nx
import pandas as pd
import numpy as np
import seaborn as sns


def get_shortest_path_likelihood(env, start, end, paths=None):
    """
    Return the probability of following any the shortest-length paths from start to end
    """
    path_log_likelihood = []
    path_probas = []
    for path in (paths or nx.all_shortest_paths(env.graph, start, end)):
        task = env._generate_task(end)
        node = env.population[start]
        for step_ix in path[1:]:
            try:
                softmaxes = learners._exp_over_sumexp(task.features, node.w_container)
                path_log_likelihood.append(np.log(softmaxes[step_ix]))
            except AttributeError:
                path_log_likelihood.append(np.log(1./len(node.adjacencies)))
            node = env.population[step_ix]
        path_probas.append(np.exp(sum(path_log_likelihood)))
        path_log_likelihood = []
    return sum(path_probas), len(path)


def get_dyad_data(env, dyads):
    """
    Create a dictionary of per-dyad information used in performance monitoring
    """
    dyad_data = {}
    for dyad in dyads:
        data_dict = {}
        data_dict.update({'start_' + k: v for k, v in get_attrs(env, dyad[0]).items()})
        data_dict.update({'end_' + k: v for k, v in get_attrs(env, dyad[1]).items()})
        data_dict['shortest_paths'] = list(nx.all_shortest_paths(env.graph, *dyad))
        data_dict['shortest_path_length'] = len(data_dict['shortest_paths'][0])
        dyad_data[dyad] = data_dict
    return dyad_data


def get_env_likelihood_samples(env, as_df=True, n_tasks=36000, sample_each=400, n_dyads=1000):
    dyads = get_dyads(env, target_len=n_dyads)
    dyad_data = get_dyad_data(env, dyads)
    likelihood_samples = []
    env.debug = False
    env.show = False
    for i in range(n_tasks):
        env.initiate_task()
        env.run_task()

        if i % sample_each == 0:
            if as_df:
                df = path_likelihood_with_dyad_traits(env, dyads, dyad_data)
                df['time'] = i
                likelihood_samples.append(df)
            else:
                likelihood_samples.append(
                    [get_shortest_path_likelihood(env, *dyad) for dyad in dyads]
                )
    if as_df:
        return pd.concat(likelihood_samples)
    else:
        return likelihood_samples


def get_attrs(pop, ix):
    """
    Get features from node ix in pop.

    :param pop: an Environment
    :param ix: a node label
    :return: key-value pairs of the node's features
    :rtype: dict

    :Example:

    >>> from abm import analysis, nxpops, io
    >>> cfg = io.ConfigReader('../setup.json').get_config()
    >>> pop = nxpops.SoftmaxNxEnvironment(**cfg)
    >>> analysis.get_attrs(pop, 3)
    {u'color': u'blue', u'region': u'east'}

    .. note:: the keys in the returned dict are read from pop.attributes
    .. seealso:: :func:`get_dyad_data`
    .. warning:: ix must be a valid node index
    """
    return {key: pop.population[ix][key] for key in pop.attributes}


def path_likelihood_with_dyad_traits(env, dyads, dyad_data):
    data = []
    for dyad in dyads:
        li, plen = get_shortest_path_likelihood(
            env, *dyad, paths=dyad_data[dyad]['shortest_paths']
        )
        learnt_over_best = learnt_over_shortest_path_len(
            env, *dyad, shortest_len=dyad_data[dyad]['shortest_path_length']
        )
        data_dict = {'li': li, 'plen': plen, 'learnt_over_best': learnt_over_best}
        data_dict.update(dyad_data[dyad])
        data.append(data_dict)
    return pd.DataFrame(data)


def get_dyads(env, target_len=1000):
    dyads = []
    while len(dyads) < target_len:
        pair = env._pick_start_end()
        try:
            d = nx.shortest_path_length(env.graph, *pair)
        except nx.NetworkXNoPath:
            continue
        if d > 1:
            dyads.append(tuple(pair))
    return dyads


def learnt_over_shortest_path_len(env, start, end, shortest_len=None):
    """
    Return the ratio of learnt likeliest path to shortest path len
    """

    shortest_len = shortest_len or nx.shortest_path_length(env.graph, start, end)

    learnt_len = 0

    task = env._generate_task(start, end)
    node = env.population[start]
    while node.index != end and learnt_len < 40:
        learnt_len += 1
        try:
            softmaxes = learners._exp_over_sumexp(task.features, node.w_container)
            argmax = max(softmaxes.iteritems(), key=operator.itemgetter(1))[0]
        except AttributeError:
            argmax = random.choice(node.adjacencies)
        node = env.population[argmax]
    return learnt_len / float(shortest_len)


def _group_sample_by_time(samples, key='li'):
    # make a list of lists, then turn into 2d ndarray
    return np.array(
        list(samples.groupby('time').apply(lambda df: df[key].tolist()).values)
    )


def segment_learning_df(df, prefixes=['start_', 'end_']):
    """
    Separates dyad learning stats df based on whether dyad members
    have matching / nonmatching / partially matching attributes
    :param prefixes: specify the column prefixes for dyad attributes
    returns {segment_name: dataframe slice}
    """
    attrs = reduce(
        lambda a, b: a.intersection(b),
        [set([c[len(p):] for c in df.columns if c.startswith(p)]) for p in prefixes]
    )

    # make boolean vectors of where each attr match/don't match
    matches = [
        reduce(operator.eq, [df[p + attr] for p in prefixes])
        for attr in attrs
    ]
    return dict(
        full_mismatch=df[reduce(lambda a, b: ~a & ~b, matches)],
        full_match=df[reduce(operator.and_, matches)],
        some_mismatch=df[reduce(operator.xor, matches)]
    )


def plot_segment_stats(segments,
                       segment_keys=['full_mismatch', 'full_match', 'some_mismatch'],
                       colors=['red', 'blue', 'green'],
                       measure_key='li'):
    for seg_key, color in zip(segment_keys, colors):
        grouped_time_samples = _group_sample_by_time(segments[seg_key], key=measure_key)
        sns.tsplot(data=grouped_time_samples.T, color=color)


def plot_learning_df(df, key='li', **kwargs):
    segments = segment_learning_df(df)
    segment_keys = [k for k, v in kwargs.items() if v]
    plot_segment_stats(segments, measure_key=key, segment_keys=segment_keys)
    return segments
