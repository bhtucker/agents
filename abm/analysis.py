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


def get_shortest_path_likelihood(env, start, end):
    """
    Return the probability of following any the shortest-length paths from start to end
    """
    path_log_likelihood = []
    try:
        for path in nx.all_shortest_paths(env.graph, start, end):
            task = env._generate_task(end)
            node = env.population[start]
            for step_ix in path[1:]:
                try:
                    softmaxes = learners._exp_over_sumexp(task.features, node.w_container)
                    path_log_likelihood.append(np.log(softmaxes[step_ix]))
                except AttributeError:
                    path_log_likelihood.append(np.log(1./len(node.adjacencies)))
                node = env.population[step_ix]
    except nx.NetworkXNoPath:
        return 0., 0
    return np.exp(sum(path_log_likelihood)), len(path)


def get_dyad_data(env, dyads):
    dyad_data = {}
    for dyad in dyads:
        data_dict = {}
        data_dict.update({'start_' + k: v for k, v in get_attrs(env, dyad[0]).items()})
        data_dict.update({'end_' + k: v for k, v in get_attrs(env, dyad[1]).items()})
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
    return {key: pop.population[ix][key] for key in pop.attributes}


def path_likelihood_with_dyad_traits(env, dyads, dyad_data):
    data = []
    for dyad in dyads:
        li, plen = get_shortest_path_likelihood(env, *dyad)
        learnt_over_best = learnt_over_shortest_path_len(env, *dyad)
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


def learnt_over_shortest_path_len(env, start, end):
    """
    Return the ratio of learnt likeliest path to shortest path len
    """
    try:
        shortest_len = nx.shortest_path_length(env.graph, start, end)
    except nx.NetworkXNoPath:
        return 0.

    learnt_len = 0

    task = env._generate_task(end)
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


def plot_learning_df(df, full_mismatch=True, no_mismatch=True, some_mismatch=True, key='li'):
    attrs = set([c.lstrip('start_') for c in df.columns if c.startswith('start_')]).intersection(
        set([c.lstrip('end_') for c in df.columns if c.startswith('end_')])
    )

    mismatches = [df['start_' + t] != df['end_' + t] for t in attrs]
    matches = [df['start_' + t] == df['end_' + t] for t in attrs]
    if full_mismatch:
        full_mismatch_df = df[reduce(operator.and_, mismatches)]
        full_mismatch_ll_samples = _group_sample_by_time(full_mismatch_df, key=key)
        sns.tsplot(data=full_mismatch_ll_samples.T, color='red')
    if no_mismatch:
        full_match_df = df[reduce(operator.and_, matches)]
        full_match_ll_samples = _group_sample_by_time(full_match_df, key=key)
        sns.tsplot(data=full_match_ll_samples.T, color='blue')
    if some_mismatch:
        some_mismatch_df = df[reduce(operator.or_, matches)]
        some_mismatch_ll_samples = _group_sample_by_time(some_mismatch_df, key=key)
        sns.tsplot(data=some_mismatch_ll_samples.T, color='green')
