'''
Run Hotspotter on a set of already-named aids and calculate the mAP, "top-k" accuracy, 
the CMC curve for name ranking and for annotation ranking. These aids, as set below, are 
all the valid aids, but this can be changed.  The version of Hotspotter run here caches a 
significant amount of visualization data, so it is a bit slow.  A TO DO is to find a faster 
version. Another issue is that it currently does not run in parallel.  It should be easy to
fix, but I have not made the time.
​
Some notes:
​
1. Annotations that are singletons - have no other annotations with the same name -- are
discovered and ignored in the statistical calculations
​
2. CMC values are produced for two forms of scoring:
   a. name scoring, whene matching results are aggregated across all annotations with the
      same name on a keypoint-by-keypoint
   b. annotation scoring, where matches are not aggregated across other annotations.
   When there is only one correct match for a query annotation then the name score and the
   annotation score are the same --- but the rank could be different
​
3. Results are gathered for each annotation as a query. Thus, for example, if there are 
   20 anotations for one name and 2 for another then the first annotaion with be represented 
   10x more in the results.
​
​
For background on mAP, see
   https://towardsdatascience.com/breaking-down-mean-average-precision-map-ae462f623a52
stopping after section 2 because we aren't working on object detection here.
​
'''

def rank(ibs, result, cm_key=None):
    """
    For a single query result from Hotspotter, compute and return:
    1. The ordered ranks of the annotations that are correct matches.
    2. The number of possible correct matches.
    3. The rank of the query name according to HotSpotter's name scoring.
    4. List of database matches and corresponding scores for the query. 
    """
    cm_dict = result['cm_dict']
    if cm_key is None:
        cm_key = list(cm_dict.keys())
        assert len(cm_key) == 1
        cm_key = cm_key[0]
    cm = cm_dict[cm_key]

    # Get the name and the nid
    query_name = cm['qname']
    qnid = ibs.get_name_rowids_from_text(query_name)

    # Get the ordered list of pairs of scores and the database nids that they camer from
    annot_uuid_list = cm['dannot_uuid_list']
    annot_score_list = cm['annot_score_list']
    daid_list = ibs.get_annot_aids_from_uuid(annot_uuid_list)
    dnid_list = ibs.get_annot_nids(daid_list)
    daid_score_list = sorted(zip(daid_list, annot_score_list), reverse=True)
    dscore_list = sorted(zip(annot_score_list, dnid_list), reverse=True)

    #  Make the list of the annotation ranks of correct matches.
    annot_ranks = []
    for rank, (dscore, dnid) in enumerate(dscore_list):
        if dnid == qnid:
            annot_ranks.append(rank)   # Ranks will need to start at 1, not 0, but we will handle this later
    
    #  Find the number of possible correct matches for the query.
    num_for_qnid = len(ibs.get_name_aids(qnid)) - 1  # This is the number of possible matches for the query 

    # Make a sorted list of name scores and the associated name ids
    name_list = cm['unique_name_list']
    name_score_list = cm['name_score_list']
    dnid_list = ibs.get_name_rowids_from_text(name_list)
    dscore_list = sorted(zip(name_score_list, dnid_list), reverse=True)

    # Find the rank of the query name in this list. Even though this is returned as a list it is
    # actually just empty or has a single entry.
    name_ranks = []
    for rank, (dscore, dnid) in enumerate(dscore_list):
        if dnid == qnid:
            name_ranks.append(rank)
            break

    return annot_ranks, num_for_qnid, name_ranks, daid_score_list  
    
# HotSpotter Query Config
query_config = {
    # HotSpotter LNBNN
    'K': 5,
    'Knorm': 5,

    # HotSpotter Background Subtraction
    'fg_on': True,
    'prescore_method': 'csum',

    # HotSpotter Spatial Verification
    'sv_on': True,

    # HotSpotter Aggregation
    'can_match_sameimg': False,
    'can_match_samename': True,
    'score_method': 'csum',
}

# Define qaids and daids here:
qaids = []
daids = []

'''
Run HotSpotter. 
'''

# Run HotSpotter query, sending all qaids at once to query_chips_graph.
query_result = ibs.query_chips_graph(
    qaid_list=open_qaids,
    daid_list=daids,
    query_config_dict=query_config,
    echo_query_params=False 
)

# Keep a list of the AP values, one for each qaid that has at least one correct match in the daids
all_AP = []

# Count the number of singletons qaids. 
num_singletons = 0

# Set the max rank to consider.  With HS CMC flattens out quickly, it does not have
# to be very hight
MAX_RANK = 12   # Should be  at least 10

#  Keep track of the counts of the top name rank and the top annotation rank
name_rank_counts = [0] * (MAX_RANK + 1)
annot_rank_counts = [0] * (MAX_RANK + 1)
qaid_top_scores= dict()

# Calculate AP and rank for each qaid
cm_dict = query_result['cm_dict']
cm_keys = list(cm_dict.keys())

for cm_key in cm_keys:  # If we run sequentially, as above, len(cm_keys) should be 1.
    cm = cm_dict[cm_key]              # Result dictionary
    qannot_uuid = cm['qannot_uuid']   # Recording of results is by uuid not qaid since qaid is internal
    qaid = ibs.get_annot_aids_from_uuid(qannot_uuid)

    # Get the indivdual annotation ranks, the number of possible matches, and the top rank
    # according to name scoring.
    annot_ranks, num_for_qnid, name_ranks, score_list = rank(ibs, query_result, cm_key=cm_key)
    qaid_top_scores[qaid] = score_list

    # If there are no other annotations with the same nid as the query, then don't contribute to the stats
    if num_for_qnid == 0:
        num_singletons += 1
        continue

    # Compute the Precision@i values for the ranked annotations.
    # As an example, for annot_ranks = [0, 3, 5, 8] this should produce
    #    precisions = [1, 0.5, 0.5, 0.444]
    # Note that we never actually compute the Precision@i values for ranks that aren't correct matches
    precisions = [(i + 1) / (r + 1) for i, r in enumerate(annot_ranks)]
    max_possible = min(MAX_RANK, num_for_qnid)
    if len(precisions) > max_possible:
        precisions = precisions[:max_possible]
    AP = sum(precisions)  / max_possible
    all_AP.append(AP)

    #  Record the name rank in the historggram
    if len(name_ranks) == 0 or name_ranks[0] >= MAX_RANK:
        name_rank_counts[MAX_RANK] += 1
    else:
        nr = name_ranks[0]
        name_rank_counts[nr] += 1

    # Record the annotation rank in the histogram.
    if len(annot_ranks) == 0 or annot_ranks[0] >= MAX_RANK:
        annot_rank_counts[MAX_RANK] += 1
    else:
        ar = annot_ranks[0]
        annot_rank_counts[nr] += 1

# Get mAP
mAP = np.mean(all_AP)
print(f'\nAt the end\n\nmAP {mAP:.4}')

""" From the histogram of name annotranks, compute the CMC curves """
cmc_name = np.array(name_rank_counts).cumsum() / sum(name_rank_counts)
cmc_annot = np.array(annot_rank_counts).cumsum() / sum(annot_rank_counts)

print(f'\nCMC by Name and by Annot:\n')
for i, (cn, cr) in enumerate(zip(cmc_name, cmc_annot)):
    print(f'{i+1:2}: {cn:.4}, {cr:.4}')