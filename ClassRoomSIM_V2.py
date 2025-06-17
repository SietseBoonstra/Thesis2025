import numpy as np
import pandas as pd
import random
import math
import matplotlib.pyplot as plt
from collections import Counter
from scipy.stats import norm
import seaborn as sns

# ———— Configuration ————
CFG = {
    'setup':   {'mean':60.0,  'sd':30.0},
    'assign':  {'mean':5.0,   'sd':2.5},
    'form':    {'low':20.0,   'high':40.0},
    'speed': {
        'small_tutorial': {'mean': 0.8,  'sd': 0.4},
        'large_lecture':  {'mean': 0.5,  'sd': 0.25},
        'workshop':       {'mean': 1.2,  'sd': 0.6}
    },
    'group_sz': 4,
    'heterogeneity_weights': {
        'numeric': 2/3,
        'topic':   1/3,
    },
    'visualization_sample_size': 30,
    'pilot_reps': 50,
    'epsilon': 0.1,
    'alpha': 0.05,
    'K': 1
}

SCENARIOS = {
    'small_tutorial': range(10,  31,  5),
    'large_lecture':  range(30, 101, 10),
    'workshop':       range(15,  46,  5),
}

ROOMS = {
    'small_tutorial': {'rows':3,  'area':(6.0,4.0)},
    'large_lecture':  {'rows':10, 'area':(20.0,15.0)},
    'workshop':       {'rows':None,'area':(10.0,10.0)},
}

TOPICS = ['A','B','C','D']
TOPIC_MAP = {t: [int(i == j) for j in range(len(TOPICS))] for i, t in enumerate(TOPICS)}

# ———— Distance Modeling ————
def compute_group_travel_time(df, scenario):
    def compute_distance(x1, y1, x2, y2):
        if scenario == 'workshop':
            return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
        elif scenario == 'small_tutorial':
            return np.abs(x1 - x2) + np.abs(y1 - y2)
        elif scenario == 'large_lecture':
            return x1 + x2 + np.abs(y1 - y2)
        else:
            return np.abs(x1 - x2) + np.abs(y1 - y2)

    def find_best_anchor(group_df):
        min_total_time = float('inf')
        best_idx = None
        for idx, anchor in group_df.iterrows():
            total_time = 0
            for _, stu in group_df.iterrows():
                dist = compute_distance(stu['x'], stu['y'], anchor['x'], anchor['y'])
                travel_time = dist / stu['speed']
                total_time += travel_time
            if total_time < min_total_time:
                min_total_time = total_time
                best_idx = idx
        return best_idx

    df['T_find'] = 0.0
    for gid, group_df in df.groupby('group_id'):
        best_idx = find_best_anchor(group_df)
        anchor = group_df.loc[best_idx]
        t_find = []
        for idx, stu in group_df.iterrows():
            dist = compute_distance(stu['x'], stu['y'], anchor['x'], anchor['y'])
            travel_time = dist / stu['speed']
            t_find.append(travel_time)
        max_time = max(t_find)
        df.loc[group_df.index, 'T_find'] = max_time

    return df

# ———— Helpers ————
def make_seats(n, room):
    w,h = room['area']; rows=room['rows']
    if rows:
        cols = math.ceil(n/rows)
        xs = np.linspace(0,w,cols); ys = np.linspace(0,h,rows)
        pts = [(x,y) for y in ys for x in xs]
        return pts[:n]
    xs = np.random.uniform(0,w,n); ys = np.random.uniform(0,h,n)
    return list(zip(xs,ys))

def pairwise_similarity(s1, s2):
    v1 = np.array([s1['motivation'], s1['preparation']] + TOPIC_MAP[s1['topic']])
    v2 = np.array([s2['motivation'], s2['preparation']] + TOPIC_MAP[s2['topic']])
    return -np.linalg.norm(v1 - v2)

def similarity_grouping(students, group_size=4, group_sizes=None, homogeneous=True):
    """
    Forms groups by similarity/dissimilarity, supporting either a fixed group_size (old) or a list of group_sizes (preferred).
    """
    n = len(students)
    sim = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            s = pairwise_similarity(students[i], students[j])
            sim[i, j] = s
            sim[j, i] = s

    unassigned = set(range(n))
    groups = []

    # Decide group sizes
    if group_sizes is not None:
        sizes = group_sizes.copy()
    else:
        n_full = n // group_size
        remainder = n % group_size
        sizes = [group_size] * n_full
        if remainder > 0:
            sizes.append(remainder)

    for sz in sizes:
        if not unassigned:
            break
        # Start with the pair with max/min similarity
        pairs = [(i, j, sim[i, j]) for i in unassigned for j in unassigned if i < j]
        if not pairs:
            leftover = list(unassigned)
            groups.append([students[idx] for idx in leftover])
            unassigned = set()
            break

        key = (lambda x: x[2]) if homogeneous else (lambda x: -x[2])
        i0, j0, _ = max(pairs, key=key)
        group_idxs = {i0, j0}
        unassigned.remove(i0)
        unassigned.remove(j0)

        while len(group_idxs) < sz and unassigned:
            scores = []
            for u in unassigned:
                avg_sim = np.mean([sim[u, g] for g in group_idxs])
                scores.append((u, avg_sim))
            u_pick, _ = max(scores, key=(lambda x: x[1]) if homogeneous else (lambda x: -x[1]))
            group_idxs.add(u_pick)
            unassigned.remove(u_pick)

        groups.append([students[i] for i in group_idxs])

    if unassigned:
        groups.append([students[i] for i in unassigned])

    return groups

def seed_cluster(students, group_size=4, group_sizes=None, homogeneous=True, topic_map=TOPIC_MAP):
    """
    Assign students to groups based on similarity, supporting either a fixed group_size or a list of group_sizes.
    If group_sizes is provided, it overrides group_size.
    """
    unassigned = students.copy()
    groups = []

    def fv(s):
        return np.array([s['motivation'], s['preparation']] + topic_map[s['topic']])

    # Decide sizes for each group
    if group_sizes is None:
        # Default: all groups have group_size, except possibly the last one
        n = len(students)
        n_full = n // group_size
        remainder = n % group_size
        sizes = [group_size] * n_full
        if remainder > 0:
            sizes.append(remainder)
    else:
        sizes = group_sizes.copy()

    for sz in sizes:
        if not unassigned:
            break
        if len(unassigned) <= sz:
            groups.append(unassigned.copy())
            break

        seed = random.choice(unassigned)
        unassigned.remove(seed)
        if not unassigned:
            groups.append([seed])
            break

        fv0 = fv(seed)
        dists = [(stu, np.linalg.norm(fv0 - fv(stu))) for stu in unassigned]
        dists.sort(key=lambda x: x[1], reverse=not homogeneous)
        chosen = [stu for stu, _ in dists[:sz-1]]
        group = [seed] + chosen
        for stu in chosen:
            unassigned.remove(stu)
        groups.append(group)

    return groups

def compute_group_sizes(n_students, group_size):
    """Returns a list with the number of students in each group, favoring full-sized groups and using smaller ones as needed."""
    n_full = n_students // group_size
    remainder = n_students % group_size
    if remainder == 0:
        sizes = [group_size] * n_full
    elif remainder == 1 and group_size > 2:
        # Split remainder: e.g., for 25 and 4, do 4*4 + 3*3
        sizes = [group_size] * (n_full - 1) + [group_size - 1] * 3
    elif remainder == 2 and group_size > 2:
        sizes = [group_size] * (n_full - 2) + [group_size - 1] * 2
    elif remainder == 3 and group_size > 2:
        sizes = [group_size] * (n_full - 1) + [group_size - 1]
    else:
        # Fallback: one smaller group for leftover students
        sizes = [group_size] * n_full + [remainder]
    return sizes

def random_group_ids(n, group_sz=None, group_sizes=None):
    """
    Returns an array of group IDs for each student.
    If group_sizes is given, it is used; otherwise, use fixed group_sz (old behavior).
    """
    perm = np.random.permutation(n)
    gids = np.empty(n, int)
    idx = 0
    gid = 0

    if group_sizes is not None:
        for sz in group_sizes:
            for _ in range(sz):
                gids[perm[idx]] = gid
                idx += 1
            gid += 1
    else:
        for i, student_idx in enumerate(perm):
            gids[student_idx] = i // group_sz
    return gids

def numeric_heterogeneity(g):
    vals = np.concatenate([g['motivation'].values, g['preparation'].values])
    return np.sqrt(np.mean((vals - vals.mean())**2))

def topic_heterogeneity(ts):
    freqs = np.array(list(Counter(ts).values()), float)
    ps = freqs / freqs.sum()
    return 1 - np.sum(ps**2)

def normalize_std(std_val, scale_min=1, scale_max=5):
    max_std = (scale_max - scale_min) / 2
    return std_val / max_std if max_std != 0 else 0

def normalize_blau(blau_val, k):
    max_blau = 1 - 1/k if k > 1 else 0
    return blau_val / max_blau if max_blau != 0 else 0

def simulate(n, scenario, process, cfg, rooms, seed=None):
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    seats = make_seats(n, rooms[scenario])
    df = pd.DataFrame({
        'student_id':   np.arange(n),
        'motivation':   np.random.randint(1,6,n),
        'preparation':  np.random.randint(1,6,n),
        'topic':        np.random.choice(TOPICS,n),
        'speed':        np.clip(
                            np.random.normal(cfg['speed'][scenario]['mean'], cfg['speed'][scenario]['sd'], n),
                            0.1, None
                        )
    })
    df[['x','y']] = pd.DataFrame(seats, index=df.index)

    if process in ('new', 'sim'):
        T_setup = max(0, np.random.normal(cfg['setup']['mean'], cfg['setup']['sd']))
        df['T_form_each'] = np.random.uniform(cfg['form']['low'], cfg['form']['high'], n)
    else:
        T_setup = 0.0
        assign_sp = max(0, np.random.normal(cfg['assign']['mean'], cfg['assign']['sd']))
        df['T_form_each'] = assign_sp

    if process == 'new':
        students = df.to_dict('records')
        group_sizes = compute_group_sizes(len(students), 4)
        groups = seed_cluster(students, group_sizes=group_sizes, homogeneous=True, topic_map=TOPIC_MAP)
        gid_map = {student['student_id']: gid for gid, grp in enumerate(groups) for student in grp}
        df['group_id'] = df['student_id'].map(gid_map)
    elif process == 'sim':
        students = df.to_dict('records')
        group_sizes = compute_group_sizes(len(students), 4)
        groups = similarity_grouping(students, group_sizes=group_sizes, homogeneous=True)
        gid_map = {student['student_id']: gid for gid, grp in enumerate(groups) for student in grp}
        df['group_id'] = df['student_id'].map(gid_map)
    else:
        df['group_id'] = random_group_ids(n, cfg['group_sz'])

    df['T_setup'] = T_setup
    df = compute_group_travel_time(df, scenario)
    df['T_total'] = df['T_setup'] + df['T_form_each'] + df['T_find']

    df['scenario']   = scenario
    df['process']    = process
    df['n_students'] = n

    if process in ('new', 'sim'):
        T_form = df['T_form_each'].max()
        T_move = df['T_find'].max()
        T_grouping = T_setup + T_form + T_move
    else:
        T_move = df['T_find'].max()
        T_grouping = assign_sp * n + T_move

    df['T_grouping'] = T_grouping
    return df



# --- Estimate N_required from pilot (this section is fine) ---

pilot_reps = 10
alpha = CFG['alpha']
epsilon = CFG['epsilon']
K = CFG['K']
z_star = norm.ppf(1 - alpha / (2 * K))
deltas = []

for rep in range(pilot_reps):
    for scenario, sizes in SCENARIOS.items():
        for n in sizes:
            # Use different random seeds for old and new to ensure diversity
            old_seed = rep * 100000 + n * 1000 + 1
            new_seed = rep * 100000 + n * 1000 + 2
            df_old = simulate(n, scenario, 'old', CFG, ROOMS, seed=old_seed)
            df_new = simulate(n, scenario, 'new', CFG, ROOMS, seed=new_seed)

            old_grp = df_old.groupby('group_id').agg(
                H_num=('motivation', lambda x: normalize_std(numeric_heterogeneity(df_old.loc[x.index]))),
                H_top=('topic', lambda x: normalize_blau(topic_heterogeneity(x), k=4)),
                T_grouping=('T_grouping', 'first')
            ).reset_index()
            new_grp = df_new.groupby('group_id').agg(
                H_num=('motivation', lambda x: normalize_std(numeric_heterogeneity(df_new.loc[x.index]))),
                H_top=('topic', lambda x: normalize_blau(topic_heterogeneity(x), k=4)),
                T_grouping=('T_grouping', 'first')
            ).reset_index()
            weight_h_top = CFG['heterogeneity_weights']['topic']
            weight_h_num = CFG['heterogeneity_weights']['numeric']
            old_het = weight_h_top * old_grp['H_top'].mean() + weight_h_num * old_grp['H_num'].mean()
            new_het = weight_h_top * new_grp['H_top'].mean() + weight_h_num * new_grp['H_num'].mean()
            diff = new_het - old_het
            deltas.append(diff)

sigma_hat = np.std(deltas, ddof=1)
N_required = int(np.ceil((z_star * sigma_hat / epsilon) ** 2))
print(f"N_required: {N_required}")

# --- Main simulation ---
all_results = []
all_results_time = []

for scenario, sizes in SCENARIOS.items():
    for n in sizes:
        for process in ('old', 'new', 'sim'):
            for rep in range(max(N_required, 100)):
                df = simulate(n, scenario, process, CFG, ROOMS, seed=None)
                T_grouping = df['T_grouping'].iloc[0]
                grp = df.groupby('group_id')

                # Apply crowding factor if needed
                baseline_map = {'small_tutorial': 10, 'workshop': 15, 'large_lecture': 30}
                baseline = baseline_map.get(scenario, 10)
                gamma = 0.1
                crowding_factor = 1 + gamma * (len(df) - baseline) / baseline
                df['T_find'] *= crowding_factor

                # Heterogeneity tracking
                df_het = grp.agg(
                    H_num=('motivation', lambda x: normalize_std(numeric_heterogeneity(df.loc[x.index]))),
                    H_top=('topic', lambda x: normalize_blau(topic_heterogeneity(x), k=4)),
                    T_grouping=('T_grouping', 'first')
                ).reset_index()
                weight_h_top = CFG['heterogeneity_weights']['topic']
                weight_h_num = CFG['heterogeneity_weights']['numeric']
                df_het['H_overall'] = weight_h_top * df_het['H_top'] + weight_h_num * df_het['H_num']
                df_het['scenario'] = scenario
                df_het['process'] = process
                df_het['n_students'] = n
                df_het['rep'] = rep
                all_results.append(df_het)

                # Time breakdown
                if process == 'old':
                    T_assign = df['T_form_each'].iloc[0]
                    T_assign_total = T_assign * len(df)
                    T_form_max = 0.0
                else:
                    T_assign_total = 0.0
                    T_form_max = df['T_form_each'].max()
                T_setup = df['T_setup'].iloc[0]
                T_find_max = df['T_find'].max()
                all_results_time.append(pd.DataFrame([{
                    'T_setup': T_setup,
                    'T_assign': T_assign_total,
                    'T_form': T_form_max,
                    'T_find': T_find_max,
                    'T_grouping': T_grouping,
                    'process': process,
                    'scenario': scenario,
                    'n_students': n,
                    'rep': rep
                }]))

dfg = pd.concat(all_results, ignore_index=True)
df_time = pd.concat(all_results_time, ignore_index=True)

# --- 4. Stacked bar plot for specific scenario/class size ---
scenario_filter = 'large_lecture'
n_filter = 80
df_case = df_time[
    (df_time['scenario'] == scenario_filter) &
    (df_time['n_students'] == n_filter)
]
time_components = (
    df_case
    .groupby('process')[['T_setup', 'T_assign', 'T_form', 'T_find']]
    .mean()
    .reset_index()
)
time_components.set_index('process')[['T_setup', 'T_assign', 'T_form', 'T_find']].plot(
    kind='bar', stacked=True, figsize=(8, 5), rot=0
)
plt.title(f"Average Grouping Time Components — {scenario_filter.replace('_',' ').title()} (n = {n_filter})")
plt.ylabel("Time (seconds)")
plt.xlabel("Grouping Method")
plt.grid(True)
plt.tight_layout()
plt.show()

# Example settings
scenario = 'large_lecture'
n = 25
seed = 123  # fixed for reproducibility

# Simulate both grouping methods
df_old = simulate(n, scenario, 'old', CFG, ROOMS, seed=seed)
df_new = simulate(n, scenario, 'new', CFG, ROOMS, seed=seed)

def group_table(df):
    rows = []
    for gid, group in df.groupby('group_id'):
        for idx, row in group.iterrows():
            rows.append({
                'Group': int(gid),
                'Student': int(row['student_id']),
                'Motivation': int(row['motivation']),
                'Preparation': int(row['preparation']),
                'Topic': row['topic']
            })
    return pd.DataFrame(rows)

tbl_old = group_table(df_old)
tbl_new = group_table(df_new)

# Sort and re-index each table
tbl_old = tbl_old.sort_values(['Group', 'Student']).reset_index(drop=True)
tbl_new = tbl_new.sort_values(['Group', 'Student']).reset_index(drop=True)

# Pad to equal length
maxlen = max(len(tbl_old), len(tbl_new))
tbl_old = tbl_old.reindex(range(maxlen))
tbl_new = tbl_new.reindex(range(maxlen))

# Rename columns to show method
tbl_old.columns = [f'Old_{col}' for col in tbl_old.columns]
tbl_new.columns = [f'New_{col}' for col in tbl_new.columns]

# Concatenate for Excel
tbl_compare = pd.concat([tbl_old, tbl_new], axis=1)

tbl_compare.to_excel('group_comparison_example.xlsx', index=False)
print("Excel export done! -> group_comparison_example.xlsx")

# Export heterogeneity results for each scenario as separate CSV files for STATA

for scenario_name in dfg['scenario'].unique():
    # Filter data for the scenario
    scenario_df = dfg[dfg['scenario'] == scenario_name].copy()
    scenario_df = scenario_df.rename(columns={
        'n_students': 'NStudents',
        'H_overall': 'OverallHeterogeneity',
        'process': 'Process',
        'scenario': 'Scenario',
        'rep': 'Rep'
    })
    export_cols = ['Process', 'Scenario', 'NStudents', 'OverallHeterogeneity', 'Rep']
    scenario_df = scenario_df[export_cols]
    # Make filename readable
    filename = f"heterogeneity_{scenario_name}.csv"
    scenario_df.to_csv(filename, index=False)
    print(f"Exported {filename}")

for scenario_name in df_time['scenario'].unique():
    scenario_time_df = df_time[df_time['scenario'] == scenario_name].copy()
    scenario_time_df = scenario_time_df.rename(columns={
        'n_students': 'NStudents',
        'T_grouping': 'GroupingTime',
        'process': 'Process',
        'scenario': 'Scenario',
        'rep': 'Rep'
    })
    export_cols = ['Process', 'Scenario', 'NStudents', 'GroupingTime', 'Rep']
    scenario_time_df = scenario_time_df[export_cols]
    filename = f"grouping_time_{scenario_name}.csv"
    scenario_time_df.to_csv(filename, index=False)
    print(f"Exported {filename}")

# --- Main simulation: Heterogeneous grouping (same procedure as homogeneous) ---
all_results_hetero = []
all_results_time_hetero = []

for scenario, sizes in SCENARIOS.items():
    for n in sizes:
        for process in ('hetero_new', 'hetero_sim'):
            for rep in range(max(N_required, 100)):
                # Simulate class (same as before, but use custom grouping below)
                df = simulate(n, scenario, 'new' if process == 'hetero_new' else 'sim', CFG, ROOMS, seed=None)

                # Overwrite group assignment with heterogeneous grouping
                students = df.to_dict('records')
                if process == 'hetero_new':
                    group_sizes = compute_group_sizes(len(students), 4)
                    groups = seed_cluster(students, group_sizes=group_sizes, homogeneous=False, topic_map=TOPIC_MAP)
                elif process == 'hetero_sim':
                    group_sizes = compute_group_sizes(len(students), 4)
                    groups = similarity_grouping(students, group_sizes=group_sizes, homogeneous=False)
                gid_map = {student['student_id']: gid for gid, grp in enumerate(groups) for student in grp}
                df['group_id'] = df['student_id'].map(gid_map)

                # Recompute group-based features using new groupings
                df = compute_group_travel_time(df, scenario)

                T_grouping = df['T_grouping'].iloc[0]
                grp = df.groupby('group_id')

                # Apply crowding factor if needed (identical to above)
                baseline_map = {'small_tutorial': 10, 'workshop': 15, 'large_lecture': 30}
                baseline = baseline_map.get(scenario, 10)
                gamma = 0.1
                crowding_factor = 1 + gamma * (len(df) - baseline) / baseline
                df['T_find'] *= crowding_factor

                # Heterogeneity tracking
                df_het = grp.agg(
                    H_num=('motivation', lambda x: normalize_std(numeric_heterogeneity(df.loc[x.index]))),
                    H_top=('topic', lambda x: normalize_blau(topic_heterogeneity(x), k=4)),
                    T_grouping=('T_grouping', 'first')
                ).reset_index()
                weight_h_top = CFG['heterogeneity_weights']['topic']
                weight_h_num = CFG['heterogeneity_weights']['numeric']
                df_het['H_overall'] = weight_h_top * df_het['H_top'] + weight_h_num * df_het['H_num']
                df_het['scenario'] = scenario
                df_het['process'] = process
                df_het['n_students'] = n
                df_het['rep'] = rep
                all_results_hetero.append(df_het)

                # Time breakdown
                T_setup = df['T_setup'].iloc[0]
                if process == 'hetero_new' or process == 'hetero_sim':
                    T_assign_total = 0.0
                    T_form_max = df['T_form_each'].max()
                else:
                    T_assign = df['T_form_each'].iloc[0]
                    T_assign_total = T_assign * len(df)
                    T_form_max = 0.0
                T_find_max = df['T_find'].max()
                all_results_time_hetero.append(pd.DataFrame([{
                    'T_setup': T_setup,
                    'T_assign': T_assign_total,
                    'T_form': T_form_max,
                    'T_find': T_find_max,
                    'T_grouping': T_grouping,
                    'process': process,
                    'scenario': scenario,
                    'n_students': n,
                    'rep': rep
                }]))

dfg_hetero = pd.concat(all_results_hetero, ignore_index=True)
df_time_hetero = pd.concat(all_results_time_hetero, ignore_index=True)

process_map = {
    'old': 'Random',
    'sim': 'Similarity (Homogeneous)',
    'hetero_sim': 'Similarity (Heterogeneous)',
    'new': 'Seed-Cluster (homogeneous)',
    'hetero_new': 'Seed-Cluster (heterogeneous)',
    'seed-cluster': 'SeedCluster',
}

# Export per scenario, adding 'old' process from homogeneous CSVs
for scenario_name in SCENARIOS.keys():
    # Heterogeneity
    dfg_hetero_sub = dfg_hetero[dfg_hetero['scenario'] == scenario_name].copy()
    dfg_hetero_sub = dfg_hetero_sub.rename(columns={
        'n_students': 'NStudents',
        'H_overall': 'OverallHeterogeneity',
        'process': 'Process',
        'scenario': 'Scenario',
        'rep': 'Rep'
    })
    # Get 'old' from homogeneous results for this scenario
    dfg_old = dfg[(dfg['scenario'] == scenario_name) & (dfg['process'] == 'old')].copy()
    dfg_old = dfg_old.rename(columns={
        'n_students': 'NStudents',
        'H_overall': 'OverallHeterogeneity',
        'process': 'Process',
        'scenario': 'Scenario',
        'rep': 'Rep'
    })
    # Optionally keep only relevant columns
    het_cols = ['Process', 'Scenario', 'NStudents', 'OverallHeterogeneity', 'Rep']
    dfg_hetero_out = pd.concat([dfg_old[het_cols], dfg_hetero_sub[het_cols]], ignore_index=True)
    het_outfile = f"heterogeneity_heterogeneous_with_manual_{scenario_name}.csv"
    dfg_hetero_out.to_csv(het_outfile, index=False)
    print(f"Exported {het_outfile}")

    # Grouping time
    df_time_hetero_sub = df_time_hetero[df_time_hetero['scenario'] == scenario_name].copy()
    df_time_hetero_sub = df_time_hetero_sub.rename(columns={
        'n_students': 'NStudents',
        'T_grouping': 'GroupingTime',
        'process': 'Process',
        'scenario': 'Scenario',
        'rep': 'Rep'
    })
    df_time_old = df_time[(df_time['scenario'] == scenario_name) & (df_time['process'] == 'old')].copy()
    df_time_old = df_time_old.rename(columns={
        'n_students': 'NStudents',
        'T_grouping': 'GroupingTime',
        'process': 'Process',
        'scenario': 'Scenario',
        'rep': 'Rep'
    })
    time_cols = ['Process', 'Scenario', 'NStudents', 'GroupingTime', 'Rep']
    df_time_hetero_out = pd.concat([df_time_old[time_cols], df_time_hetero_sub[time_cols]], ignore_index=True)
    time_outfile = f"grouping_time_heterogeneous_with_manual_{scenario_name}.csv"
    df_time_hetero_out.to_csv(time_outfile, index=False)
    print(f"Exported {time_outfile}")

dfg_both = pd.concat([dfg, dfg_hetero], ignore_index=True)
df_time_both = pd.concat([df_time, df_time_hetero], ignore_index=True)

# Composite heterogeneity for all methods and groupings
for scenario in dfg_both['scenario'].unique():
    plt.figure(figsize=(9,6))
    subset = dfg_both[dfg_both['scenario'] == scenario]
    d_agg_het = (
        subset.groupby(['n_students', 'process'])['H_overall']
        .mean().reset_index()
    )
    # Define marker/color map
    color_map = {
        'old': 'tab:blue', 'new': 'tab:orange', 'sim': 'tab:green',
        'hetero_new': 'tab:orange', 'hetero_sim': 'tab:green'
    }
    marker_map = {
        'old': 'o', 'new': 's', 'sim': '^',
        'hetero_new': 's', 'hetero_sim': '^'
    }
    linestyle_map = {
        'old': '-', 'new': '-', 'sim': '-',
        'hetero_new': '--', 'hetero_sim': '--'
    }
    process_labels = {
        'old': 'Random', 'new': 'Seed-Cluster (Homogeneous)', 'sim': 'Similarity (Homogeneous)',
        'hetero_new': 'Seed-Cluster (Heterogeneous)', 'hetero_sim': 'Similarity (Heterogeneous)'
    }
    for process in d_agg_het['process'].unique():
        d = d_agg_het[d_agg_het['process'] == process]
        plt.plot(
            d['n_students'], d['H_overall'],
            marker=marker_map[process], color=color_map[process],
            linestyle=linestyle_map[process],
            label=process_labels.get(process, process)
        )
    plt.title(f"Composite Heterogeneity — {scenario.replace('_',' ').title()}")
    plt.xlabel("Class Size")
    plt.ylabel("H_overall")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

for scenario in df_time_both['scenario'].unique():
    plt.figure(figsize=(9,6))
    subset = df_time_both[df_time_both['scenario'] == scenario]
    d_agg_time = (
        subset.groupby(['n_students', 'process'])['T_grouping']
        .mean().reset_index()
    )
    # Use the same maps as above
    for process in d_agg_time['process'].unique():
        d = d_agg_time[d_agg_time['process'] == process]
        plt.plot(
            d['n_students'], d['T_grouping'],
            marker=marker_map[process], color=color_map[process],
            linestyle=linestyle_map[process],
            label=process_labels.get(process, process)
        )
    plt.title(f"Total Grouping Time — {scenario.replace('_',' ').title()}")
    plt.xlabel("Class Size")
    plt.ylabel("Total Grouping Time (seconds)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Concatenate homogeneous and heterogeneous results (after renaming 'process' as above)
dfg_box = pd.concat([dfg, dfg_hetero], ignore_index=True)
dfg_box['process'] = dfg_box['process'].map(process_map)

# --- Heterogeneity boxplot for each scenario ---
for scenario_name in dfg_box['scenario'].unique():
    plt.figure(figsize=(10,6))
    subset = dfg_box[dfg_box['scenario'] == scenario_name]
    sns.boxplot(data=subset, x='process', y='H_overall', showmeans=True)
    plt.title(f"Total Heterogeneity by Process\n({scenario_name.replace('_',' ').title()})")
    plt.xlabel("Process")
    plt.ylabel("Total Heterogeneity")
    plt.tight_layout()
    plt.show()

for scenario_name in dfg_box['scenario'].unique():
    plt.figure(figsize=(14,6))
    subset = dfg_box[dfg_box['scenario'] == scenario_name]
    sns.boxplot(data=subset, x='n_students', y='H_overall', hue='process', showmeans=True)
    plt.title(f"Total Heterogeneity by Class Size and Process\n({scenario_name.replace('_',' ').title()})")
    plt.xlabel("Class Size")
    plt.ylabel("Total Heterogeneity")
    plt.legend(title="Process")
    plt.tight_layout()
    plt.show()

df_time_box = pd.concat([df_time, df_time_hetero], ignore_index=True)
df_time_box['process'] = df_time_box['process'].map(process_map)

# By scenario
for scenario_name in df_time_box['scenario'].unique():
    plt.figure(figsize=(10,6))
    subset = df_time_box[df_time_box['scenario'] == scenario_name]
    sns.boxplot(data=subset, x='process', y='T_grouping', showmeans=True)
    plt.title(f"Grouping Time by Process\n({scenario_name.replace('_',' ').title()})")
    plt.xlabel("Process")
    plt.ylabel("Grouping Time (seconds)")
    plt.tight_layout()
    plt.show()

for scenario_name in df_time_box['scenario'].unique():
    plt.figure(figsize=(14,6))
    subset = df_time_box[df_time_box['scenario'] == scenario_name]
    sns.boxplot(data=subset, x='n_students', y='T_grouping', hue='process', showmeans=True)
    plt.title(f"Grouping Time by Class Size and Process\n({scenario_name.replace('_',' ').title()})")
    plt.xlabel("Class Size")
    plt.ylabel("Grouping Time (seconds)")
    plt.legend(title="Process")
    plt.tight_layout()
    plt.show()