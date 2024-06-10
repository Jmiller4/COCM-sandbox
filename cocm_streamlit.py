import streamlit as st
import pandas as pd
import random
import numpy as np


def binarize(df):
  return df.applymap(lambda x: 1 if x > 0 else 0)

def convert_to_pcts(funding_dict):
  total = sum(funding_dict[k] for k in funding_dict.keys())
  return {k : funding_dict[k] / total for k in funding_dict.keys()}


def drop_empty_projects(donation_df):

  for c in donation_df.columns:
    if donation_df[c].sum() == 0:
      donation_df.drop(c, inplace=True, axis=1)
  return donation_df


def COCM(donation_df, cluster_df, calcstyle='markov', harsh=True):
  # run CO-CM on a set of funding amounts and clusters
  # calcstyle is a variable signifying how to compute similarity scores between users and projects
  # harsh is a boolean determining how we should scale contributions, given similarity scores

  projects = donation_df.columns
  clusters = cluster_df.columns
  donors = donation_df.index
  cluster_members = cluster_df.index

  # normalize the cluster dataframe so that rows sum to 1. Now, an entry tells us the "weight" that a particular cluster has for a particular user.
  # if a user is in 0 clusters, their row will be a bunch of NaNs if we naively divide by 1.
  # we shouldn't have any such users anyways, but just in case, we'll fill such a row with 0s instead
  normalized_clusters = cluster_df.apply(lambda row: row / row.sum() if any(row) else 0, axis=1)

  binarized_clusters = binarize(cluster_df)

  assert calcstyle in ['markov', 'og', 'pct_friends']
  if calcstyle == 'markov':
    normalized_users = cluster_df.transpose().apply(lambda row: row / row.sum() if any(row) else 0, axis=1)
    PJP = normalized_users.dot(normalized_clusters)
    k_indicators = normalized_clusters.dot(PJP)
    k_indicators = k_indicators.apply(lambda row: np.maximum(row, binarized_clusters.loc[row.name]), axis=1)

  if calcstyle == 'pct_friends':
    # friendship_matrix is a matrix whose rows and columns are both wallets,
    # and a value of 1 at index i,j means that wallets i and j are in at least one cluster together.
    friendship_matrix = cluster_df.dot(cluster_df.transpose()).apply(lambda col: col > 0)

    # k_indicators is a dataframe with wallets as rows and clusters as columns.
    # if wallet i is not in cluster g, then entry i,g is is the fraction of i's friends who are in cluster g (i's friends are the agents i is in a shared cluster with).
    # if wallet i is in cluster g, then entry i,g is 1.

    # in the past, we used cluster_df in the following line instead of binarized_clusters
    k_indicators = friendship_matrix.dot(binarized_clusters).apply(lambda row: row / friendship_matrix.loc[row.name].sum(), axis=1)
    # ... and the following line used cluster_df instead of binarized_clusters
    k_indicators = k_indicators.apply(lambda row: np.maximum(row, binarized_clusters.loc[row.name]), axis=1)
  
  if calcstyle == 'og':
    # friendship_matrix is a matrix whose rows and columns are both wallets,
    # and a value greater than 0 at index i,j means that wallets i and j are in at least one group together.
    friendship_matrix = cluster_df.dot(cluster_df.transpose())

    # k_indicators is a dataframe with wallets as rows and stamps as columns.
    # entry i,g is True if wallet i is in a shared group with anyone from g, and False otherwise.
    k_indicators = friendship_matrix.dot(cluster_df).apply(lambda col: col > 0)

  # Create a dictionary to store funding amounts for each project.
  funding = {p: 0 for p in projects}

  for p in projects:
    # get the actual k values for this project using contributions and indicators.

    # C will be used to build the matrix of k values.
    # It is a matrix where rows are wallets, columns are clusters, and the ith row of the matrix just has wallet i's contribution to the project in every entry.
    C = pd.DataFrame(index=donors, columns = ['_'], data = donation_df[p].values).dot(pd.DataFrame(index= ['_'], columns = clusters, data=1))
    # C is attained by taking the matrix multiplication of the column vector donation_df[p] (which is every agent's donation to project p) and a row vector with as many columns as projects, and a 1 in every entry
    # the above line is so long mainly because you need to cast Pandas series' (i.e. vectors) as dataframes (i.e. matrices) for the matrix multiplication to work.

    # now, K is a matrix where rows are wallets, columns are projects, and entry i,g ranges between c_i and sqrt(c_i) depending on i's relationship with cluster g and whether "fancy" was set to true or not.
    K = (k_indicators * C.pow(1/2)) + ((1 - k_indicators) * C)

    if harsh == True:
      K = (1 - k_indicators) * C

    # Now we have all the k values, which are one of the items inside the innermost sum expressed in COCM.
    # the other component of these sums is a division of each k value by the number of groups that user is in.
    # P_prime is a matrix that combines k values and total group memberships to attain the value inside the aforementioned innermost sum.
    # In other words, entry g,h of P_prime is:
    #
    #       sum_{i in g} K(i,h) / T_i
    #
    # where T_i is the total number of groups that i is in
    P_prime = K.transpose().dot(normalized_clusters)

    # Now, we can create P_prime, whose non-diagonal entries g,h represent the pairwise subsidy given to the pair of groups g and h.
    P = (P_prime * P_prime.transpose()).pow(1/2)

    # The diagonal entries of P are not relevant, so get rid of them. We only care about the pairwise subsidies between distinct groups.
    np.fill_diagonal(P.values, 0)

    # Now the sum of every entry in P is the amount of subsidy funding COCM awards to the project.
    funding[p] += P.sum().sum()


  return funding

st.write("## COCM Sandbox")

st.write("### Inputs")

max_donors = 10
max_projects = 10
max_donation = 50

if 'don' not in st.session_state:

	st.session_state.don = {d: {p:0 for p in range(max_projects)} for d in range(max_donors) }

if 'def_don' not in st.session_state:

	st.session_state.def_don = {d: {p:0 for p in range(max_projects)} for d in range(max_donors) }

col1, col2 = st.columns(2)
num_donors = col1.number_input("Number of Donors", min_value=1, max_value=max_donors, value=5, step = 1)
num_projects = col2.number_input("Number of Projects", min_value=1, max_value=max_projects, value=3, step = 1)

donors = range(num_donors)
projects = range(num_projects)


for d in range(max_donors):
	for p in range(max_projects):
		if d not in donors or p not in projects:
			st.session_state.def_don[d][p] = 0


if st.button('Randomize Donations'):

	for d in range(max_donors):
		for p in range(max_projects):
			st.session_state.def_don[d][p] = 0
			if d in donors and p in projects:
				if random.random() > 0.333:
					st.session_state.def_don[d][p] = random.choice(list(range(1,max_donation+1)))

proj_cols = st.columns(num_projects)

for p in projects:
	for d in donors:
		#st.session_state['don'][d][p] = 
		st.session_state.don[d][p] = proj_cols[p].slider(f'donor {d + 1} -> project {p + 1}', min_value = 0, max_value = max_donation, value = st.session_state.def_don[d][p])


st.session_state.donation_df = pd.DataFrame(index = donors, columns = projects, data = [[st.session_state.don[d][p] for p in projects] for d in donors])

for d in donors:
	if st.session_state.donation_df.loc[d].sum() == 0:
		st.session_state.donation_df.drop(d, inplace=True)

projects_without_donors = []
for p in projects:
	if st.session_state.donation_df[p].sum() == 0:
		st.session_state.donation_df.drop(p, axis=1, inplace=True)
		projects_without_donors.append(p)

if len(projects_without_donors) == num_projects:
	res = {p: 0 for p in projects}
else:	
	res = COCM(st.session_state.donation_df , st.session_state.donation_df)

	for p in projects_without_donors:
		res[p] = 0

for p in st.session_state.donation_df.columns:
	res[p] += st.session_state.donation_df[p].sum()

st.write("### Results")

res_df = pd.DataFrame(index=[f'Project {p + 1}' for p in projects], columns = ['Amount'])
for p in projects:
	res_df.loc[f'Project {p + 1}','Amount'] = res[p]

st.table(res_df)

# st.table(res)