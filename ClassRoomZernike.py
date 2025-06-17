import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns

# ———— CONFIGURATION ————
CFG = {
    'setup':   {'mean':60.0,  'sd':30.0},      # time to explain grouping task (seconds)
    'assign':  {'mean':5.0,   'sd':2.5},       # time to assign to group (if instructor does it)
    'form':    {'low':20.0,   'high':40.0},    # self-forming group time (seconds)
    'speed': {
        'large_lecture':  {'mean': 0.15,  'sd': 0.075},  # student walking speed (m/s)
    },
    'group_sz': 4,
}

# --- REAL ROOM DATA ---
ROOMS = {
    'Donald_Smitszaal': {
        'rows': 8,
        'seats_per_row': [13]*8,
        'row_width': 7.15,           # meters
        'room_length': 7.20,         # meters (front-to-back)
        'aisle_left': 1.70,          # meters
        'aisle_right': 1.38,         # meters
        'capacity': 104
    },
    'Blauwe_Zaal': {
        'rows': 11,                  # 10 rows of 16, 1 row of 12 at front
        'seats_per_row': [16]*10 + [12],
        'row_width': 8.80,           # meters (all rows; front row slightly shorter in seats, same in width)
        'room_length': 9.35,
        'aisle_left': 1.25,
        'aisle_right': 1.20,
        'capacity': 172
    },
    '5491.0015': {
        'rows': 17,                  # 15 of 16, 1 of 14, 1 of 12
        'seats_per_row': [16]*15 + [14, 12],
        'row_width': 9.20,
        'room_length': 14.45,
        'aisle_left': 1.35,
        'aisle_right': 1.375,
        'capacity': 266
    }
}

# -------- SCENARIO HOOK ----------
SCENARIOS = {
    'large_lecture': [30, 40, 50, 60, 70, 80, 90, 100],  # choose class sizes; can cover full capacity for each room
}

# --------- Seat Arrangement (accurate by room layout) ---------
def make_seats(n, room):
    """
    Generate seat coordinates for all students based on real room layout.
    - n: number of students
    - room: room dict as in ROOMS above
    Returns: list of (x, y) positions
    """
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
        # Center the seats for shorter rows
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
    # Only assign students to existing seats
    return seat_positions[:n]

# --------- Distance Modeling ----------
def compute_group_travel_time(df, room, scenario):
    """Assigns T_find (max travel to anchor in each group) for all groups."""
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

# --------- Group Assignment (Random Only) ----------
def random_group_ids(n, group_sz):
    perm = np.random.permutation(n)
    gids = np.empty(n, int)
    for i, student_idx in enumerate(perm):
        gids[student_idx] = i // group_sz
    return gids


def plot_seat_layout(room, room_name):
    """
    Plot all seat positions for a given room.
    """
    capacity = room['capacity']
    seats = make_seats(capacity, room)
    xs, ys = zip(*seats)
    fig, ax = plt.subplots(figsize=(10, 5))

    # Plot seats
    ax.scatter(xs, ys, s=50, c='dodgerblue', edgecolors='black', zorder=2, label='Seats')

    # Draw room boundary (optionally add aisles as shaded regions)
    room_width = room['row_width'] + room['aisle_left'] + room['aisle_right']
    room_length = room['room_length']
    rect = plt.Rectangle((0, 0), room_width, room_length, linewidth=2, edgecolor='gray', facecolor='none', zorder=1)
    ax.add_patch(rect)

    # Aisles as shaded zones (optional)
    ax.axvspan(0, room['aisle_left'], color='lightgray', alpha=0.2, label='Left Aisle')
    ax.axvspan(room['aisle_left'] + room['row_width'], room_width, color='lightgray', alpha=0.2, label='Right Aisle')

    # Add some axis info
    ax.set_title(f"Seat Layout: {room_name.replace('_', ' ')}")
    ax.set_xlabel("Width (m)")
    ax.set_ylabel("Depth (m)")
    ax.set_xlim(-0.5, room_width + 0.5)
    ax.set_ylim(-0.5, room_length + 0.5)
    ax.set_aspect('equal')
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.show()

# --------- Main Simulation Function ----------
def simulate(n, scenario, room_name, process, cfg, rooms, seed=None):
    """
    Simulate grouping for n students in a given scenario and room.
    Processes:
        - 'self': students self-organize
        - 'assigned': instructor assigns groups
    """
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
    if process == 'self':
        T_setup = max(0, np.random.normal(cfg['setup']['mean'], cfg['setup']['sd']))
        df['T_form_each'] = np.random.uniform(cfg['form']['low'], cfg['form']['high'], n)
    elif process == 'assigned':
        T_setup = 0.0
        assign_sp = max(0, np.random.normal(cfg['assign']['mean'], cfg['assign']['sd']))
        df['T_form_each'] = assign_sp
    df['T_setup'] = T_setup
    df = compute_group_travel_time(df, room, scenario)
    occupancy = n / room['capacity']
    gamma = 0.4  # adjust as needed, 0.25-0.5 is realistic for big rooms
    if occupancy > 0.7:  # only apply if more than 70% full
        crowding_factor = 1 + gamma * (occupancy - 0.7) / 0.3  # linearly up to +gamma at full
        df['T_find'] *= crowding_factor
    df['T_total'] = df['T_setup'] + df['T_form_each'] + df['T_find']
    if process == 'self':
        T_form = df['T_form_each'].max()
        T_move = df['T_find'].max()
        T_grouping = T_setup + T_form + T_move
    else:
        T_move = df['T_find'].max()
        T_grouping = assign_sp * n + T_move
    # Add columns for analysis
    df['T_grouping'] = T_grouping
    df['room'] = room_name
    df['process'] = process
    df['n_students'] = n
    return df

# --------- Simulation Wrapper (per scenario, per room) ---------
def simulate_across_conditions(rooms, scenario='large_lecture', n_reps=100):
    all_results = []
    for room_name in rooms:
        for n in SCENARIOS[scenario]:
            # Only simulate up to room capacity
            if n > rooms[room_name]['capacity']:
                continue
            for process in ['self', 'assigned']:
                for rep in range(n_reps):
                    df = simulate(n, scenario, room_name, process, CFG, rooms, seed=None)
                    T_grouping = df['T_grouping'].iloc[0]
                    all_results.append({
                        'room': room_name,
                        'process': process,
                        'n_students': n,
                        'T_grouping': T_grouping,
                        'rep': rep
                    })
    return pd.DataFrame(all_results)

def plot_grouping_time_breakdown(breakdown, room_name, n, process):
    labels = []
    times = []
    colors = []
    if breakdown['T_setup'] > 0:
        labels.append('Setup')
        times.append(breakdown['T_setup'])
        colors.append('tab:blue')
    if breakdown['T_assign'] > 0:
        labels.append('Assign')
        times.append(breakdown['T_assign'])
        colors.append('tab:orange')
    if breakdown['T_form'] > 0:
        labels.append('Form')
        times.append(breakdown['T_form'])
        colors.append('tab:green')
    labels.append('Find')
    times.append(breakdown['T_find'])
    colors.append('tab:red')

    fig, ax = plt.subplots(figsize=(6, 2.5))
    ax.barh([''], times, color=colors, left=[sum(times[:i]) for i in range(len(times))])
    for i, (l, t) in enumerate(zip(labels, times)):
        ax.text(sum(times[:i]) + t/2, 0, f"{l}\n{t:.1f}s", va='center', ha='center', color='white', fontsize=10, fontweight='bold')
    ax.set_xlim(0, sum(times)*1.1)
    ax.set_yticks([])
    ax.set_title(f"Grouping Time Components\n{room_name.replace('_',' ')} (n={n}, {process})")
    ax.set_xlabel("Seconds")
    plt.tight_layout()
    plt.show()

def show_grouping_time_breakdown(n, scenario, room_name, process, CFG, ROOMS, seed=None):
    df = simulate(n, scenario, room_name, process, CFG, ROOMS, seed=seed)
    T_setup = df['T_setup'].iloc[0]
    if process == 'self':
        T_form = df['T_form_each'].max()
        T_assign = 0.0
    else:
        T_form = 0.0
        T_assign = df['T_form_each'].iloc[0] * n
    T_find = df['T_find'].max()
    T_grouping = df['T_grouping'].iloc[0]
    print(f"--- Grouping Time Breakdown for {room_name} (n={n}, process={process}) ---")
    print(f"T_setup:   {T_setup:.2f} sec")
    print(f"T_assign:  {T_assign:.2f} sec")
    print(f"T_form:    {T_form:.2f} sec")
    print(f"T_find:    {T_find:.2f} sec")
    print(f"-----------------")
    print(f"T_grouping: {T_grouping:.2f} sec (total)")
    return {
        'T_setup': T_setup,
        'T_assign': T_assign,
        'T_form': T_form,
        'T_find': T_find,
        'T_grouping': T_grouping
    }

# Example usage:
breakdown = show_grouping_time_breakdown(
    n=80,
    scenario='large_lecture',
    room_name='Blauwe_Zaal',
    process='self',
    CFG=CFG,
    ROOMS=ROOMS,
    seed=42
)


# --------- Example Usage/Plotting ---------
if __name__ == "__main__":
    df_results = simulate_across_conditions(ROOMS, scenario='large_lecture', n_reps=100)

    # After running simulate_across_conditions(...)
    df_results.to_csv('group_formation_times_all.csv', index=False)
    print("Exported all group formation times to group_formation_times_all.csv")

    # Example usage:
    plot_grouping_time_breakdown(breakdown, room_name='Blauwe_Zaal', n=80, process='self')

    # --- Boxplot: by class size and room
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

    # --- Lineplot: means ± std, by class size and room
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
    plt.title("Mean Grouping Time by Class Size and Room (All Processes Combined)")
    plt.xlabel("Class Size")
    plt.ylabel("Mean Total Grouping Time (seconds)")
    plt.legend(title="Room")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    for room_name, room in ROOMS.items():
        plot_seat_layout(room, room_name)

