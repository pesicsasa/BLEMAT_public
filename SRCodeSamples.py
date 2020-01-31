import time
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from networkx.algorithms import community
from community import community_louvain, generate_dendrogram
import networkx as nx
from datetime import datetime as dt, timedelta
from joblib import Parallel, delayed
from networkx.drawing.nx_agraph import graphviz_layout
import igraph as ig
from nltk.tokenize import word_tokenize

def build_apartments_labels():
    """Builds a list of apartment labels specific to the datasets in the project.
       Accepts: None
       Returns:  List of strings
    """
    appts = []
    apt_string = 'APT_{}_FLOOR_{}_{}'
    for i in range(1, 7):
        appts.append(apt_string.format(i, 0, 'B'))
        appts.append(apt_string.format(i, 0, 'U'))

    # 1-5
    for i in range(1, 9):
        for j in range(1, 6):
            appts.append(apt_string.format(i, j, 'B'))

    for i in range(1, 10):
        for j in range(1, 6):
            appts.append(apt_string.format(i, j, 'U'))

    # 6-7
    for i in range(1, 5):
        appts.append(apt_string.format(i, 6, 'U'))
        appts.append(apt_string.format(i, 7, 'U'))
        appts.append(apt_string.format(i, 6, 'B'))
        appts.append(apt_string.format(i, 7, 'B'))
    # 8-11
    for i in range(1, 5):
        appts.append(apt_string.format(i, 8, 'B'))
        appts.append(apt_string.format(i, 9, 'B'))
        appts.append(apt_string.format(i, 10, 'B'))
        appts.append(apt_string.format(i, 11, 'B'))
    for i in range(1, 4):
        appts.append(apt_string.format(i, 8, 'U'))
        appts.append(apt_string.format(i, 9, 'U'))
        appts.append(apt_string.format(i, 10, 'U'))
        appts.append(apt_string.format(i, 11, 'U'))

    # 12-15
    for i in range(1, 4):
        appts.append(apt_string.format(i, 12, 'B'))
        appts.append(apt_string.format(i, 13, 'B'))
        appts.append(apt_string.format(i, 14, 'B'))
        appts.append(apt_string.format(i, 15, 'B'))
    for i in range(1, 3):
        appts.append(apt_string.format(i, 12, 'U'))
        appts.append(apt_string.format(i, 13, 'U'))
        appts.append(apt_string.format(i, 14, 'U'))
        appts.append(apt_string.format(i, 15, 'U'))

    return appts

def extract_communities_girvan_newman(G):
    """Does community detection based on Girvan-Newman algorithm.
    Accepts: Networkx Graph G
    Returns:  None
    """
    communities_generator = community.girvan_newman(G)
    for c in next(communities_generator):
        print('COM: {}'.format(tuple(sorted(c))))
    dendro = generate_dendrogram(G.to_undirected())
    print('Dendrogram: {}'.format(dendro))

def extract_communities_louvain(G, highlight=False, which=1):
    """Does community detection based on Louvain method.
       Accepts: Networkx Graph G, highlight (boolean, to produce another graph that highlights a community), which (integer, which community to highlight)
       Returns:  None
    """
    H = nx.Graph()
    H.add_nodes_from(G)
    H.add_edges_from(G.edges())
    partition = community_louvain.best_partition(H)
    pos = nx.spring_layout(H)
    plt.figure(figsize=(50,50))
    nx.draw(H, pos, with_labels=True, font_size=50, node_size=8000, cmap=plt.cm.RdYlBu, node_color=list(partition.values()))
    nx.draw_networkx_edges(H, pos, alpha=0.3, edge_color='lightgray')
    plt.savefig('LM_communities.png')
    plt.close()

    if highlight:
        node_size = []
        for n, c in partition.items():
            if c == which:
                node_size.append(16000)
            else:
                partition[n] = 0
                node_size.append(8000)

        plt.figure(figsize=(50, 50))
        pos = nx.spring_layout(H)  # compute graph layout
        nx.draw(H, pos, with_labels=True, font_size=50, node_size=node_size, cmap=plt.cm.winter,
                                       node_color=list(partition.values()))
        nx.draw_networkx_edges(H, pos, alpha=0.3, edge_color='lightgray')
        plt.savefig('LM_communities_enlarged_{}.png'.format(which))
        plt.close()

def build_relationships_graph_for_building(graphs):
    """Builds a weighted graph of social relationships at the building granularity level, for all apartments' tenants.
    Accepts: List of weighted Networkx Graphs
    Returns:  Networkx Graph G (weighted social relationships graph)
    """
    buildingG = nx.DiGraph()
    for G in graphs:
        # 'Usable' apartments and edges that do not reference an 'OUT' node
        usable_apts = []
        usable_edges = []
        if G != None:
            nodes = list(G.nodes())
            for n in nodes:
                if 'OUT' in n: continue
                stays = int(n.split(':')[1].replace("[", "").replace("]",""))
                usable_apts.append(stays)
            for e in list(G.edges()):
                (fromN, toN) = e
                w = G[fromN][toN]['weight']
                if 'OUT' in fromN or 'OUT' in toN: continue
                usable_edges.append(w)

            b_edges, b_nodes = find_edges_over_weight_limit(G, 4)
            for e in b_edges:
                (fromN, toN, w) = e
                if buildingG.has_edge(fromN, toN):
                    buildingG[fromN][toN]['weight'] += w
                else:
                    buildingG.add_weighted_edges_from([e])

    labels = nx.get_edge_attributes(buildingG, 'weight')
    options = {
        'node_color': 'aquamarine',
        'node_size': 7000,
        'width': 7,
        'node_shape': 'o',
        'edge_color': 'purple',
        'font_size': 40,
        'label_pos':1
    }
    plt.figure(figsize=(40,40))
    plt.title('Building-level social dynamics', fontsize=80)
    nx.draw(buildingG, with_labels=True, pos=graphviz_layout(buildingG, prog='neato'), **options)
    nx.draw_networkx_edge_labels(buildingG, edge_labels=labels, pos=graphviz_layout(buildingG, prog='neato'), font_size=25)
    plt.savefig('building_relationships_graph.png')
    plt.close()
    return buildingG

def find_edges_over_weight_limit(G, limit):
    """Extract edges from a graph that have a weight over a certain limit
    Accepts: Networkx Graph G
    Returns:  List of edges, list of nodes
    """
    apts = build_apartments_labels()
    apts_mapping = {}
    for i in range(0, len(apts)):
        changedApt = apts[i].replace('APT', 'A')
        changedApt = changedApt.replace('FLOOR', 'F')
        apts_mapping[changedApt] = i 
    edgesF = []
    nodesF = []
    edges = list(G.edges())
    for e in edges:
        (fromN, toN) = e
        fromNStr = fromN.split("\n")[0]
        toNStr = toN.split("\n")[0]
        if 'OUT' in fromNStr or 'OUT' in toNStr: continue
        if 'F_0' in fromNStr or 'F_0' in toNStr: continue
        fromNStr = apts_mapping[fromNStr]
        toNStr = apts_mapping[toNStr]
        newe = (fromNStr, toNStr, G[fromN][toN]['weight'])
        if fromN==toN: continue
        if G[fromN][toN]['weight'] > limit:
            edgesF.append(newe)
            nodesF.append(fromNStr)
            nodesF.append(toNStr)
    return edgesF, list(set(nodesF))

def get_positions_for_beacon(beacon_mac, all_data):
    """Extracts a list of positions for a given beacon data
    Accepts: Beacon Mac address, data array
    Returns:  List of 3D positions
    """
    positions = []
    for a in all_data:
        try:
            beacon_mac_data = a['Beacons'][beacon_mac]['Location']
            positions.append(beacon_mac_data)
        except:
            continue
    return positions

def build_beacon_3d_path_graph(beacon_mac, positions):
    """Draws a 3D path graph inside building 3D model dimensions
    Accepts: Beacon Mac address, list of beacon positions
    Returns:  None
    """
    import matplotlib.pyplot as plt
    from matplotlib import style
    style.use('fivethirtyeight')
    fig = plt.figure()
    plt.title(beacon_mac)
    ax1 = fig.add_subplot(111, projection='3d')
    positions = positions[0::12]
    x = []
    y = []
    z = []
    for p in positions:
        x.append(p[0])
        y.append(p[1])
        z.append(p[2])
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)
    plt.title('{}, {} positions'.format(beacon_mac, len(positions)))
    ax1.plot(x, y, z)
    plt.savefig("{}.png".format(beacon_mac.replace(':','_')))

def aggregate_tenant_hourly_positions(beacon_mac, all_data, clean=False, clean_limit=5):
    """Extracts most common positions for a tenant/beacon for every hour of observed data
    Accepts: Beacon Mac address, beacon positioning data, clean (boolean, True if nodes that were visited less than clean_limit are to be removed from observation, False otherwise)
    Returns:  list of tenant most visited apartments per hour, number of apartments captured, number of apartments cleaned
    """
    apts_every_hour = []
    start = 0
    stop = 6
    # 1 for 10min, 6 for 1h, etc.
    while stop <= len(all_data):
        subset = all_data[start:stop]
        apts = []
        for s in subset:
            if beacon_mac in s['Beacons']:
                apt = s['Beacons'][beacon_mac]['Appartement']
                apt_trimmed = apt.replace('FLOOR', 'F')
                apt_trimmed = apt_trimmed.replace('APT', 'A')
                apts.append(apt_trimmed)
        if len(apts) == 0: apts.append('OUTSIDE')
        apts_every_hour.append(most_frequent(apts))
        start = stop
        stop = start + 6

    apts_to_remove_from_G = []
    apts_count = Counter(apts_every_hour)
    if clean:
        for a in apts_count:
            c = apts_count[a]
            if c <=clean_limit: apts_to_remove_from_G.append(a)
    return apts_every_hour, apts_count, apts_to_remove_from_G

def most_frequent(List):
    """ Helper method of aggregate_tenant_hourly_positions
       Accepts: List of apartments
       Returns:  Frequency of most visited apartment
   """
    counter = 0
    num = List[0]
    for i in List:
        curr_frequency = List.count(i)
        if (curr_frequency > counter):
            counter = curr_frequency
            num = i
    return num

def build_tenants_daily_path_graph(beacon_mac, apts, day):
    """ Builds a linear path graph of 24 nodes, one for each hour of the day with tenants most frequent position aggregated for that hour
        Accepts: Tenant's beacon Mac address, list of apartments, number of observed day in a week (1-7)
        Returns:  NetworkX Graph H
    """
    G = nx.path_graph(len(apts))
    color_map =[]
    mapping = {}
    color_map.append('palegreen')
    pos = {}
    for i in range (0,24):
        newapt = apts[i]
        if 'OUTSIDE' in newapt: newapt = newapt.replace('OUTSIDE', 'OUT')
        step_str = str(int(newapt.split('_')[len(newapt.split('_'))-1].split('S')[1]))+'-'
        newapt= step_str+''.join(newapt.split('_')[0:len(newapt.split('_'))-1])
        mapping[i] = newapt
        if i != 0: color_map.append('peachpuff')
        pos[newapt] = (1,i)
    H = nx.relabel_nodes(G, mapping)

    options = {
        'node_size': 600,
        'width': 2,
        'node_shape': 'o',
        'edge_color': 'orange',
        'font_size': 12
    }
    plt.figure(figsize=(2,15))
    plt.title('Path of Tenant with beacon {}'.format(beacon_mac))
    nx.draw(H, with_labels=True, pos=pos, node_color = color_map, **options)
    plt.savefig("paths_1week_dec/{}.png".format(beacon_mac.replace(':','_')+"_"+str(day)))
    plt.close()
    return H

def build_relationships_graph(beacon_mac, apts, apts_to_remove_from_G, node_labels=None, day=None):
    """ Builds a weighted DiGraph of social relationships for given beacon and apartments
        Accepts: Tenant's beacon Mac address, list of apartments, list of apartments to clean, new node labels (if there is need to relabel), and the day number in week (1-7, used when generating periodic path graphs)
        Returns:  NetworkX Graph G
    """
    G = nx.DiGraph()
    counter = Counter(apts)
    newapts = []
    for a in apts:
        if not a in apts_to_remove_from_G:
            newa = a + "\nStays:[{}]".format(counter[a])
            if not node_labels:
                if not 'F_1' in newa:
                    newapts.append(newa)
            else: newapts.append(newa)
    apts = newapts
    G.add_nodes_from(apts)
    if len(list(G.nodes())) == 2:
        return None
    for x in range(0, len(apts)-1):
        G.add_weighted_edges_from([(apts[x], apts[x+1], 1)])
    for x in range(0, len(apts) - 1):
        if G[apts[x]][apts[x + 1]]['weight'] >0:
            G[apts[x]][apts[x + 1]]['weight'] = G[apts[x]][apts[x + 1]]['weight']+1

    pos = None
    labels = None
    H = None

    if node_labels:
        mapping = {}
        for apt in newapts:
            aptStr = apt.split('\n')[0]
            stays = apt.split('\n')[1]
            mapping[apt] = next((s for s in node_labels if aptStr in s), None)+'\n{}'.format(stays)
        H = nx.relabel_nodes(G, mapping, copy=False)
        pos = nx.kamada_kawai_layout(H)
        labels = nx.get_edge_attributes(H, 'weight')
    else:
        pos = nx.circular_layout(G)
        labels = nx.get_edge_attributes(G, 'weight')

    options = {
        'node_color': 'aquamarine',
        'node_size': 2000,
        'width': 2,
        'node_shape':'o',
        'edge_color':'red',
        'font_size':30
    }
    if len(labels)>1:
        plt.figure(figsize=(30,30))
        plt.title('Behaviour graph for beacon: {}'.format(beacon_mac), fontsize=40)
        if H:
            nx.draw(H, pos=pos, with_labels=True, **options)
            nx.draw_networkx_edge_labels(H, pos=pos, edge_labels=labels, font_size=30)
        else:
            nx.draw(G, pos=pos, with_labels=True,**options)
            nx.draw_networkx_edge_labels(G, pos=pos, edge_labels=labels,font_size=30)
        if day:
            plt.savefig(beacon_mac.replace(":", "_") + "_day{}.png".format(day))
        else:
            plt.savefig('path_'+beacon_mac.replace(":","_")+".png")
        if H: return H
        else: return G
    else: return None

def build_tenant_weekly_path_graphs(beacon_mac, apts):
    """ Populates an list of beacon path graphs, each graph having 24 nodes (1 day path)
       Accepts: Tenant's beacon Mac address, list of apartments
       Returns:  List of NetworkX Graph H
   """
    start = 0
    stop =24
    # stopping point is 24h, 2-day graph would need to stop at 48h, etc.
    day = 1
    graphs = []
    apt_mappings = {}
    while stop <= len(apts):
        day_apts_no_steps = apts[start:stop]
        day_apts_with_steps = []
        step = 1
        for d in day_apts_no_steps:
            if d in apt_mappings:
                apt_mappings[d] +=1
            else:
                apt_mappings[d] = 1
            day_apts_with_steps.append(d+'_S{}'.format(step))
            step+=1
        G = build_tenants_daily_path_graph(beacon_mac, day_apts_with_steps, day=day)
        graphs.append(G)
        start = stop
        stop = start + 24
        day +=1
    return graphs

def generate_beacon_daily_graphs(beacon_mac, apts):
    """ Populates an list of beacon relationship graphs
       Accepts: Tenant's beacon Mac address, list of apartments
       Returns:  List of NetworkX Graphs G
   """
    start = 0
    stop =23
    day = 1
    graphs = []
    while stop < len(apts):
        day_apts_no_steps = apts[start:stop]
        day_apts_with_steps = {}
        newNodes = []
        step = 1
        for d in day_apts_no_steps:
            if d not in day_apts_with_steps: day_apts_with_steps[d] = []
            day_apts_with_steps[d].append(step)
            step+=1
        for d in day_apts_with_steps:
            range = printr(range_extract(day_apts_with_steps[d]))
            newNode = '{}\nSteps:[{}]'.format(d, range)
            newNodes.append(newNode)
        G = build_relationships_graph(beacon_mac, day_apts_no_steps, [], newNodes, day)
        graphs.append(G)
        start = stop
        stop = start + 23
        day +=1
    return graphs

def range_extract(lst):
    """ Helper method of generate_beacon_daily_graphs
       Accepts: List of numbers
       Returns:  None
   """
    lenlst = len(lst)
    i = 0
    while i < lenlst:
        low = lst[i]
        while i < lenlst - 1 and lst[i] + 1 == lst[i + 1]: i += 1
        hi = lst[i]
        if hi - low >= 2:
            yield (low, hi)
        elif hi - low == 1:
            yield (low,)
            yield (hi,)
        else:
            yield (low,)
        i += 1

def printr(ranges):
    """ Helper method of generate_beacon_daily_graphs
         Accepts: List of ranges (low and high)
         Returns:  Range String, e.g. 6-9
     """
    return( ','.join( (('%i-%i' % r) if len(r) == 2 else '%i' % r)
                     for r in ranges ) )

def calculate_cosine_similarity (x, y):
    """ Calculate Cosine similarity of two sentences
        Accepts: List of words X and Y
        Returns:  Double similarity
    """

    X = ' '.join(x)
    Y = ' '.join(y)
    X_set = word_tokenize(X)
    Y_set = word_tokenize(Y)
    l1 = []
    l2 = []

    rvector = X_set.union(Y_set)
    for w in rvector:
        if w in X_set:
            l1.append(1)
        else:
            l1.append(0)
        if w in Y_set:
            l2.append(1)
        else:
            l2.append(0)
    c = 0

    for i in range(len(rvector)):
        c += l1[i] * l2[i]
    cosine = c / float((sum(l1) * sum(l2)) ** 0.5)
    return cosine

def calculate_levenshtein_distance(seq1, seq2):
    """ Calculate Levenshtein distance between two lists of words
        Accepts: List of words seq1 and seq2
        Returns:  Double similarity
    """
    size_x = len(seq1) + 1
    size_y = len(seq2) + 1
    matrix = np.zeros((size_x, size_y))
    for x in range(size_x):
        matrix[x, 0] = x
    for y in range(size_y):
        matrix[0, y] = y

    for x in range(1, size_x):
        for y in range(1, size_y):
            if seq1[x - 1] == seq2[y - 1]:
                matrix[x, y] = min(
                    matrix[x - 1, y] + 1,
                    matrix[x - 1, y - 1],
                    matrix[x, y - 1] + 1
                )
            else:
                matrix[x, y] = min(
                    matrix[x - 1, y] + 1,
                    matrix[x - 1, y - 1] + 1,
                    matrix[x, y - 1] + 1
                )
    return (matrix[size_x - 1, size_y - 1])

def calculate_behaviour_graphs_weekly_similarity(graphs, time_start):
    """ Calculate Graph-edit distance similarity and Eigenvector similarity of every consecutive graph in list of graphs to get overall weekly graphs similarity. Modify to skip weekday-to-weekend comparison.
          Accepts: List of NetworkX graphs, first timestamp
          Returns:  None
      """
    time_start = dt.strptime(time_start,  "%m/%d/%Y, %H:%M:%S") - timedelta(hours=7)
    all_graphs_ged = []
    all_graphs_eigen = []
    times = {}
    correctly_calculated = 0
    for i in range(0, len(graphs) - 1):
        try:
            times_start_str = time_start.strftime("%m/%d/%Y, %H:%M:%S")
            g1 = graphs[0]
            g2 = graphs[i+1]
            nodes_g1 = list(g1.nodes())
            nodes_g2 = list(g2.nodes())

            mapping = {}
            for n in nodes_g1:
                mapping[n] = n.split('\n')[0]
            h1 = nx.relabel_nodes(g1, mapping)
            mapping = {}
            for n in nodes_g2:
                mapping[n] = n.split('\n')[0]
            h2 = nx.relabel_nodes(g2, mapping)

            times[i] = {}
            times[i]['start_time'] = times_start_str
            times[i]['g1'] = {}
            times[i]['g2'] = {}
            times[i]['g1']['nodes'] = list(h1.nodes())
            times[i]['g2']['nodes'] = list(h2.nodes())
            times[i]['g1']['nodes_no'] = len(list(h1.nodes()))
            times[i]['g2']['nodes_no'] = len(list(h2.nodes()))

            # Calculating GED
            res = nx.graph_edit_distance(h1, h2)
            all_graphs_ged.append(res)
            times[i]['ged'] = res

            # Calculating Eigenvector similarity
            UG1 = nx.Graph()
            UG1.add_nodes_from(h1)
            w_edges = []
            for e in h1.edges():
                (f,t) = e
                e_w = (f,t,h1[f][t]['weight'])
                w_edges.append(e_w)
            UG1.add_weighted_edges_from(w_edges)
            UG2 = nx.Graph()
            UG2.add_nodes_from(h2)
            times[i]['g1']['edges'] = w_edges
            times[i]['g1']['edges_no'] = len(w_edges)
            w_edges = []
            for e in h2.edges():
                (f, t) = e
                e_w = (f, t, h2[f][t]['weight'])
                w_edges.append(e_w)
            UG2.add_weighted_edges_from(w_edges)
            times[i]['g2']['edges'] = w_edges
            laplacian1 = nx.spectrum.laplacian_spectrum(UG1)
            laplacian2 = nx.spectrum.laplacian_spectrum(UG2)
            k1 = select_k(laplacian1)
            k2 = select_k(laplacian2)
            k = min(k1, k2)
            res = sum((laplacian1[:k] - laplacian2[:k]) ** 2)
            all_graphs_eigen.append(res)
            times[i]['eigen'] = res
            correctly_calculated +=1
            time_start = time_start + timedelta(hours=24)

        except Exception as e:
            print('Exception: {}'.format(e))
            continue
    if correctly_calculated == 6:
        print('Weekly average Eigenvector Similarity:{}'.format(sum(all_graphs_eigen)/len(all_graphs_eigen)))
        print('Weekly average GraphEditDistance Similarity:{}'.format(sum(all_graphs_ged)/len(all_graphs_ged)))
        times['average_eigen'] = sum(all_graphs_eigen)/len(all_graphs_eigen)
        times['average_ged'] = sum(all_graphs_ged)/len(all_graphs_ged)
        print('Times: {}'.format(times))

        with open('behavior_graphs_similarity.json', 'w') as outfile:
            json.dump(times, outfile)

def select_k(spectrum, minimum_energy = 0.9):
    """ Helper method for calculate_behaviour_graphs_weekly_similarity
        Accepts: Spectrum as nx.spectrum.laplacian_spectrum
        Returns:  Integer k
    """
    running_total = 0.0
    total = sum(spectrum)
    if total == 0.0:
        return len(spectrum)
    for i in range(len(spectrum)):
        running_total += spectrum[i]
        if running_total / total >= minimum_energy:
            return i + 1
    return len(spectrum)

def calculate_path_graphs_weekly_similarity(daily_graphs, period=7):
    """ Calculate weekly similarity of beacon path graphs applyign percentage similarity, cosine similarity and Levenshtein similarity. Modify to skip weekday-to-weekend comparison.
          Accepts: List of NetworkX graphs
          Returns:  None
      """
    similarity = {}
    worthy_beacons_60 = []
    worthy_beacons_70 = []
    worthy_beacons_80 = []
    worthy_beacons_90 = []

    for beacon in daily_graphs:
        similarity[beacon] = {}
        similarity[beacon]
        similarity[beacon]['average_%']=0
        similarity[beacon]['average_Cosine']=0
        similarity[beacon]['average_Levenshtein']=0

        graphs = daily_graphs[beacon]
        if len(graphs) <2 : continue
        for i in range(0, len(graphs) - 1):
            similarity[beacon][i] = {}
            g1 = graphs[i]
            g2 = graphs[i + 1]
            no_step_1 = list(map(lambda x: ''.join(x.split('-')[1]), list(g1.nodes())))
            no_step_2 = list(map(lambda x: ''.join(x.split('-')[1]), list(g2.nodes())))

            res = len(set(no_step_1) & set(no_step_2)) / float(len(set(no_step_1) | set(no_step_2))) * 100
            similarity[beacon][i]['%sim'] = res
            similarity[beacon]['average_%'] += res
            res = calculate_cosine_similarity(no_step_1, no_step_2)
            similarity[beacon][i]['cosine'] = res
            similarity[beacon]['average_Cosine'] += res
            res = calculate_levenshtein_distance(no_step_1, no_step_2)
            similarity[beacon]['average_Levenshtein'] += res
            similarity[beacon][i]['levenshtein'] = res
            similarity[beacon][i]['g1'] = {}
            similarity[beacon][i]['g2'] = {}
            similarity[beacon][i]['g1']['nodes'] = list(g1.nodes())
            similarity[beacon][i]['g2']['nodes'] = list(g2.nodes())
            similarity[beacon][i]['g1']['nodes_no'] = len(list(g1.nodes()))
            similarity[beacon][i]['g2']['nodes_no'] = len(list(g2.nodes()))
            similarity[beacon][i]['beacon'] = beacon
        similarity[beacon]['average_%'] = similarity[beacon]['average_%'] / (period-1)
        similarity[beacon]['average_Cosine'] = similarity[beacon]['average_Cosine'] / (period-1)
        similarity[beacon]['average_Levenshtein'] = similarity[beacon]['average_Levenshtein'] / (period-1)
        if (similarity[beacon]['average_%'] > 60 and similarity[beacon]['average_Cosine'] > 0.6):
            worthy_beacons_60.append(beacon)
        if (similarity[beacon]['average_%'] > 70 and similarity[beacon]['average_Cosine'] > 0.7):
            worthy_beacons_70.append(beacon)
        if (similarity[beacon]['average_%'] > 80 and similarity[beacon]['average_Cosine'] > 0.8):
            worthy_beacons_80.append(beacon)
        if (similarity[beacon]['average_%'] > 90 and similarity[beacon]['average_Cosine'] > 0.9):
            worthy_beacons_90.append(beacon)
    similarity['worthy_beacons_60_num'] = len(worthy_beacons_60)
    similarity['worthy_beacons_60'] = worthy_beacons_60
    similarity['worthy_beacons_70_num'] = len(worthy_beacons_70)
    similarity['worthy_beacons_70'] = worthy_beacons_70
    similarity['worthy_beacons_80_num'] = len(worthy_beacons_80)
    similarity['worthy_beacons_80'] = worthy_beacons_80
    similarity['worthy_beacons_90_num'] = len(worthy_beacons_90)
    similarity['worthy_beacons_90'] = worthy_beacons_90
    with open('tenant_weekly_paths_similarity.json', 'w') as outfile:
        json.dump(similarity, outfile)

def check_path_graphs_validity(daily_graphs):
    """ Check if a set of daily path graphs of each beacon are valid (number of OUT nodes high)
      Accepts: Dictionary of beacons and their associated weekly paths as NetworkX Graphs G
      Returns:  Filtered Dictionary of beacons and their associated weekly paths as NetworkX Graphs G
    """
    to_delete = []
    for beacon in daily_graphs:
        graphs = daily_graphs[beacon]
        invalid_graphs = 0
        for i in range(0, len(graphs) - 1):
            g = graphs[i]
            out_nodes = 0
            for node in list(g.nodes()):
                if 'OUT' in node: out_nodes +=1
            if out_nodes >=20: invalid_graphs +=1
        if invalid_graphs >3:
            to_delete.append(beacon)
    for key in [key for key in daily_graphs if key in to_delete]: del daily_graphs[key]
    return daily_graphs

def process_beacon(beacon_mac, beacons_generated, graphs):
    """ Build relationship graph for beacon B, used in building_relationships_graph_parallel
      Accepts: Tenant's beacon mac address, list of already processed beacons, list of already built grpahs to append to
      Returns:  Filtered Dictionary of beacons and their associated weekly paths as NetworkX Graphs G
    """
    apts, apt_stays, apts_to_remove_from_G = aggregate_tenant_hourly_positions(beacon_mac, data, clean=True)
    G = build_relationships_graph(beacon_mac, apts, apts_to_remove_from_G)
    beacons_generated.append(beacon_mac)
    if G != None:
        graphs.append(G)

def extract_girvan_newman_communities_igraph(path_in, path_out):
    """ Calculate Girvan-Newman communities, draw communities graph, extract modularity and dendrogram using iGraph library
      Accepts: Path to read Graph GML file from (can be NetworkX graph), path to write the communities graph in .png format
      Returns:  None
    """
    g = ig.Graph.Read_GML(path_in)
    d = g.community_edge_betweenness()
    p = d.as_clustering()
    Q = g.modularity(p)
    print('Modularity: {}'.format(Q))
    print('Clustering: {}'.format(p))
    print('Dendrogram: {}'.format(d))   

    i = g.community_infomap()
    colors = ["#E41A1C", "#377EB8", "#4DAF4A", "#984EA3", "#FF7F00", "#50f245", "#f1fd24", "#eefadd", "#47f1b3", "#d99ad5", "#4ed58e", "#becb45", "#677402"]
    g.vs['color'] = [None]
    for clid, cluster in enumerate(i):
        print(clid, cluster)
        for member in cluster:
            g.vs[member]['color'] = colors[clid]
    g.vs['frame_width'] = 0
    ig.plot(g, path_out)

def extract_louvain_communities_igraph(path):
    """ Calculate Louvain communities, extract modularity and dendrogram using iGraph library
      Accepts: Path to read Graph GML file from (can be NetworkX graph)
      Returns:  None
    """
    g = ig.Graph.Read_GML(path)
    h = g.as_undirected(mode="collapse")
    d = h.community_multilevel()
    Q = h.modularity(d)
    print('Modularity: {}'.format(Q))
    print('Clustering: {}'.format(d))
    print('Dendrogram: {}'.format(d))

def building_relationships_graph_parallel(path):
    """ Paralelizes building social relationship graphs for entire dataset
      Accepts: Path to positioning data
      Returns:  None
    """
    with open(path) as json_file:
        data = json.load(json_file)
        graphs = []
        beacons_generated = []
        for a in data:
            Parallel(n_jobs=-1)(delayed(process_beacon)(beacon_mac, beacons_generated, graphs) for beacon_mac in a['Beacons'])
        buildingG = build_relationships_graph_for_building(graphs)
        return buildingG

def run_all(path):
    """ Method used for testing all functionalities
      Accepts: Path to positioning data
      Returns:  None
    """
    with open(path) as json_file:
        data = json.load(json_file)
        time_start = data[0]['Timestamp']
        drawn = 0
        graphs = []
        beacons_generated = []
        daily_paths_per_beacon = {}
        for a in data:
            for beacon_mac in a['Beacons']:
                try:
                    if beacon_mac in beacons_generated:
                        continue
                    pos = get_positions_for_beacon(beacon_mac, data)
                    build_beacon_3d_path_graph(beacon_mac, pos)
                    apts, apt_stays, apts_to_remove_from_G = aggregate_tenant_hourly_positions(beacon_mac, data, clean=True)
                    G = build_relationships_graph(beacon_mac, apts, apts_to_remove_from_G)
                    daily_graphs = build_tenant_weekly_path_graphs(beacon_mac, apts)
                    daily_graphs = generate_beacon_daily_graphs(beacon_mac, apts)
                    calculate_behaviour_graphs_weekly_similarity(daily_graphs, time_start)
                    daily_paths_per_beacon[beacon_mac] = daily_graphs
                    beacons_generated.append(beacon_mac)
                    if G != None:
                        drawn+=1
                        graphs.append(G)
                except Exception as e:
                    raise(e)
                    continue
    print('Size of beacons generated: {}'.format(len(beacons_generated)))
    print('Size of daily_paths_per_beacon before filter: {}'.format(len(daily_paths_per_beacon)))
    daily_paths_per_beacon = check_path_graphs_validity(daily_paths_per_beacon)
    print('Size of daily_paths_per_beacon after filter: {}'.format(len(daily_paths_per_beacon)))
    calculate_path_graphs_weekly_similarity(daily_paths_per_beacon, period=14)
    print('It took', time.time() - start, 'seconds.')
    print("Beacons generated len: {}".format(len(beacons_generated)))
    print("Graphs len: {}".format(len(graphs)))
    buildingG = build_relationships_graph_for_building(graphs)
    nx.write_gml(buildingG, 'building_soc_rel_dec.gml')
    print("Nodes {}, edges {}".format(len(buildingG.nodes()), len(buildingG.edges())))
    extract_communities_girvan_newman(buildingG)
    extract_communities_louvain(buildingG, True, 1)

if __name__ == "__main__":
    run_all('4weeks_data_dec_2019.json')