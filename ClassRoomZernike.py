import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
from scipy.stats import norm

# ———— CONFIGURATION ————
CFG = {
    'setup':   {'mean': 60.0,  'sd': 30.0},
    'assign':  {'mean': 5.0,   'sd': 2.5},
    'form':    {'low': 20.0,   'high': 40.0},
    'speed': {
        'large_lecture':  {'mean': 0.15,  'sd': 0.075},
    },
    'group_sz': 4,
}

ROOMS = {
    'Donald_Smitszaal': {
        'rows': 8,
        'seats_per_row': [13]*8,
        'row_width': 7.15,
        'room_length': 7.20,
        'aisle_left': 1.70,
        'aisle_right': 1.38,
        'capacity': 104
    },
    'Blauwe_Zaal': {
        'rows': 11,
        'seats_per_row': [16]*10 + [12],
        'row_width': 8.80,
        'room_length': 9.35,
        'aisle_left': 1.25,
        'aisle_right': 1.20,
        'capacity': 172
    },
    '5491.0015': {
        'rows': 17,
        'seats_per_row': [16]*15 + [14, 12],
        'row_width': 9.20,
        'room_length': 14.45,
        'aisle_left': 1.35,
        'aisle_right': 1.375,
        'capacity': 266
    }
}

SCENARIOS = {
    'large_lecture': [30, 40, 50, 60, 70, 80, 90, 100],
}

def make_seats(n, room):
    if isinstance(room['seats_per_row'], int):
        seats_per_row = [room['seats_per_row']] * room['rows']
    else:
        seats_per_row = room['seats_per_row']
    rows = len(seats_per_row)
    length = room['room_length']
    row_width = room['row_width']
    y_coords = np.linspace(0, length, rows)
    seat_positions = []
    max_seats_in_row = max(seats_per_row)
    for i, seats in enumerate(seats_per_row):
        extra_space = (max_seats_in_row - seats) / max_seats_in_row * row_width / 2
        x_start = room['aisle_left'] + extra_space
        x_end = room['aisle_left'] + row_width - extra_space
        if seats > 1:
            x_coords = np.linspace(x_start, x_end, seats)
        else:
            x_coords = [(x_start + x_end) / 2.0]
        y = y_coords[i]
        for x in x_coords:
            seat_positions.append((x, y))
    return seat_positions[:n]

def compute_group_travel_time(df, room, scenario):
    def compute_distance(x1, y1, x2, y2):
        if scenario == 'large_lecture':
            left_aisle = 0
            right_aisle = room['aisle_left'] + room['row_width']
            to_left_aisle = abs(x1 - left_aisle)
            to_right_aisle = abs(x1 - right_aisle)
            exit1 = min(to_left_aisle, to_right_aisle)
            to_left_aisle_anchor = abs(x2 - left_aisle)
            to_right_aisle_anchor = abs(x2 - right_aisle)
            exit2 = min(to_left_aisle_anchor, to_right_aisle_anchor)
            aisle_travel = abs(y1 - y2)
            return exit1 + aisle_travel + exit2
        else:
            return abs(x1 - x2) + abs(y1 - y2)
    def find_best_anchor(group_df):
        min_total_time = float('inf')
        best_idx = None
        for idx, anchor in group_df.iterrows():
            total_time = sum(
                compute_distance(stu['x'], stu['y'], anchor['x'], anchor['y']) / stu['speed']
                for _, stu in group_df.iterrows()
            )
            if total_time < min_total_time:
                min_total_time = total_time
                best_idx = idx
        return best_idx
    df['T_find'] = 0.0
    for gid, group_df in df.groupby('group_id'):
        best_idx = find_best_anchor(group_df)
        anchor = group_df.loc[best_idx]
        t_find = []
        for _, stu in group_df.iterrows():
            dist = compute_distance(stu['x'], stu['y'], anchor['x'], anchor['y'])
            travel_time = dist / stu['speed']
            t_find.append(travel_time)
        df.loc[group_df.index, 'T_find'] = max(t_find)
    return df

def random_group_ids(n, group_sz):
    perm = np.random.permutation(n)
    gids = np.empty(n, int)
    for i, student_idx in enumerate(perm):
        gids[student_idx] = i // group_sz
    return gids

def simulatezernike(n, scenario, room_name, process, cfg, rooms, seed=None):
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    room = rooms[room_name]
    seats = make_seats(n, room)
    speeds = np.clip(
        np.random.normal(cfg['speed']['large_lecture']['mean'], cfg['speed']['large_lecture']['sd'], n),
        0.1, None
    )
    df = pd.DataFrame({
        'student_id': np.arange(n),
        'speed': speeds,
    })
    df[['x', 'y']] = pd.DataFrame(seats, index=df.index)
    group_sz = cfg['group_sz']
    df['group_id'] = random_group_ids(n, group_sz)
    if process == 'designed':
        T_setup = max(0, np.random.normal(cfg['setup']['mean'], cfg['setup']['sd']))
        df['T_form_each'] = np.random.uniform(cfg['form']['low'], cfg['form']['high'], n)
    elif process == 'random':
        T_setup = 0.0
        assign_sp = max(0, np.random.normal(cfg['assign']['mean'], cfg['assign']['sd']))
        df['T_form_each'] = assign_sp
    df['T_setup'] = T_setup
    df = compute_group_travel_time(df, room, scenario)
    baseline = 30
    gamma = 0.1
    crowding_factor = 1 + gamma * (len(df) - baseline) / baseline
    df['T_find'] *= crowding_factor
    df['T_total'] = df['T_setup'] + df['T_form_each'] + df['T_find']
    if process == 'designed':
        T_form = df['T_form_each'].max()
        T_move = df['T_find'].max()
        T_grouping = T_setup + T_form + T_move
    else:
        T_move = df['T_find'].max()
        T_grouping = assign_sp * n + T_move
    df['T_grouping'] = T_grouping
    df['room'] = room_name
    df['process'] = process
    df['n_students'] = n
    return df

def simulate_across_conditions(rooms, scenario='large_lecture', n_reps=100):
    all_results = []
    for room_name in rooms:
        for n in SCENARIOS[scenario]:
            if n > rooms[room_name]['capacity']:
                continue
            for process in ['designed', 'random']:
                for rep in range(n_reps):
                    df = simulatezernike(n, scenario, room_name, process, CFG, rooms, seed=None)
                    T_grouping = df['T_grouping'].iloc[0]
                    all_results.append({
                        'room': room_name,
                        'process': process,
                        'n_students': n,
                        'T_grouping': T_grouping,
                        'rep': rep
                    })
    return pd.DataFrame(all_results)

def plot_seat_layout(room, room_name):
    capacity = room['capacity']
    seats = make_seats(capacity, room)
    xs, ys = zip(*seats)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(xs, ys, s=50, c='dodgerblue', edgecolors='black', zorder=2, label='Seats')
    room_width = room['row_width'] + room['aisle_left'] + room['aisle_right']
    room_length = room['room_length']
    rect = plt.Rectangle((0, 0), room_width, room_length, linewidth=2, edgecolor='gray', facecolor='none', zorder=1)
    ax.add_patch(rect)
    ax.axvspan(0, room['aisle_left'], color='lightgray', alpha=0.2, label='Left Aisle')
    ax.axvspan(room['aisle_left'] + room['row_width'], room_width, color='lightgray', alpha=0.2, label='Right Aisle')
    ax.set_title(f"Seat Layout: {room_name.replace('_', ' ')}")
    ax.set_xlabel("Width (m)")
    ax.set_ylabel("Depth (m)")
    ax.set_xlim(-0.5, room_width + 0.5)
    ax.set_ylim(-0.5, room_length + 0.5)
    ax.set_aspect('equal')
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    rooms_to_compare = ['Blauwe_Zaal', 'Donald_Smitszaal', '5491.0015']
    class_sizes = SCENARIOS['large_lecture']
    pilot_reps = 50
    alpha = 0.05
    epsilon = 5.0
    z_star = norm.ppf(1 - alpha / 2)
    process = 'designed'
    scenario = 'large_lecture'
    ultimate_N_required = 0
    all_pairwise_stats = []

    # 1. Pilot to determine required replications
    for n in class_sizes:
        pilot_T = {room: [] for room in rooms_to_compare}
        for rep in range(pilot_reps):
            for room in rooms_to_compare:
                df = simulatezernike(
                    n=n,
                    scenario=scenario,
                    room_name=room,
                    process=process,
                    cfg=CFG,
                    rooms=ROOMS,
                    seed=rep * 1000 + hash(room) % 10000 + n
                )
                pilot_T[room].append(df['T_grouping'].iloc[0])
        for r1, r2 in itertools.combinations(rooms_to_compare, 2):
            diffs = np.array(pilot_T[r2]) - np.array(pilot_T[r1])
            sigma_hat = np.std(diffs, ddof=1)
            N_required = int(np.ceil((z_star * sigma_hat / epsilon) ** 2))
            ultimate_N_required = max(ultimate_N_required, N_required)
            all_pairwise_stats.append({
                'n': n,
                'room1': r1,
                'room2': r2,
                'sigma_hat': sigma_hat,
                'N_required': N_required,
                'mean_diff': np.mean(diffs)
            })

    print(f"\nUltimate N_required to power all pairwise comparisons, across all class sizes: {ultimate_N_required}")

    # 2. Main simulation across conditions
    df_results = simulate_across_conditions(ROOMS, scenario='large_lecture', n_reps=ultimate_N_required)
    df_results.to_csv('group_formation_times_all.csv', index=False)
    print("Exported all group formation times to group_formation_times_all.csv")

    # 3. Plotting (example with seat layouts)
    for room_name, room in ROOMS.items():
        plot_seat_layout(room, room_name)

    # 4. Boxplot: by class size and room
    plt.figure(figsize=(12, 6))
    sns.boxplot(
        data=df_results,
        x='n_students',
        y='T_grouping',
        hue='room',
        showmeans=True
    )
    plt.title("Grouping Time Distribution by Class Size and Room (All Processes Combined)")
    plt.xlabel("Class Size")
    plt.ylabel("Total Grouping Time (seconds)")
    plt.legend(title="Room")
    plt.tight_layout()
    plt.show()

    # 5. Lineplot: means by class size and room
    agg = df_results.groupby(['room', 'n_students'])['T_grouping'].mean().reset_index()
    plt.figure(figsize=(10, 6))
    for room in agg['room'].unique():
        d = agg[agg['room'] == room]
        plt.plot(
            d['n_students'],
            d['T_grouping'],
            marker='o',
            label=f"{room}"
        )
    plt.axhline(y=300, color='black', linewidth=3, linestyle='--')
    plt.title("Mean Grouping Time by Class Size and Room (All Processes Combined)")
    plt.xlabel("Class Size")
    plt.ylabel("Mean Total Grouping Time (seconds)")
    plt.legend(title="Room")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
