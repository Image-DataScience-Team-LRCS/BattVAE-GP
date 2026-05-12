import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import os


def interpolate_cycle(group, step_size):
    """Interpolates time, voltage, and current for a single cycle group."""
    time_original = group["Time"].values
    voltage_original = group["Voltage"].values
    current_original = group["Current"].values

    # Filter out NaNs
    valid_mask = (
        ~np.isnan(time_original)
        & ~np.isnan(voltage_original)
        & ~np.isnan(current_original)
    )
    time_original = time_original[valid_mask]
    voltage_original = voltage_original[valid_mask]
    current_original = current_original[valid_mask]

    # Sort by time
    sort_idx = np.argsort(time_original)
    time_original = time_original[sort_idx]
    voltage_original = voltage_original[sort_idx]
    current_original = current_original[sort_idx]

    # Remove duplicate time points
    _, unique_indices = np.unique(time_original, return_index=True)
    time_original = time_original[unique_indices]
    voltage_original = voltage_original[unique_indices]
    current_original = current_original[unique_indices]

    # Create uniform time grid
    time_new = np.arange(
        time_original.min(), time_original.max() + step_size, step_size
    )

    try:
        voltage_interp_func = interp1d(
            time_original,
            voltage_original,
            kind="linear",
            bounds_error=False,
            fill_value="extrapolate",
        )
        current_interp_func = interp1d(
            time_original,
            current_original,
            kind="nearest",
            bounds_error=False,
            fill_value=(current_original[0], current_original[-1]),
        )

        voltage_new = voltage_interp_func(time_new)
        current_new = current_interp_func(time_new)

    except Exception as e:
        print(f"Interpolation failed for cycle {group['Cycle'].iloc[0]}: {e}")
        return None

    return pd.DataFrame(
        {
            "Time": time_new,
            "Current": current_new,
            "Voltage": voltage_new,
            "Cycle": group["Cycle"].iloc[0],
        }
    )


def interpolate_all_cycles(file_path, step_size=15.0):
    """Interpolates all cycles in the given CSV file."""
    df = pd.read_csv(file_path)
    required_cols = {"Time", "Voltage", "Current", "Cycle"}
    assert required_cols.issubset(
        df.columns
    ), f"Missing required columns: {required_cols - set(df.columns)}"

    interpolated_dfs = []
    for cycle, group in df.groupby("Cycle"):
        result = interpolate_cycle(group, step_size)
        if result is not None:
            interpolated_dfs.append(result)

    if not interpolated_dfs:
        raise ValueError("No valid cycles found for interpolation.")

    df_interp = pd.concat(interpolated_dfs, ignore_index=True)
    df_interp["dt"] = df_interp.groupby("Cycle")["Time"].diff().fillna(step_size)

    return df_interp


def main():
    file = input("Enter the input CSV filename (e.g., 'time_data_C_0.5.csv'): ").strip()
    if not os.path.isfile(file):
        print("❌ File not found.")
        return

    step_size = 15.0
    df_interpolated = interpolate_all_cycles(file, step_size)

    print("\n✅ Interpolation complete.")
    print("Unique time delta values:", np.round(df_interpolated["dt"].unique(), 5))
    print("\nCycle lengths (post-interpolation):")
    print(df_interpolated.groupby("Cycle")["Voltage"].count().sort_values())

    # Optionally remove Cycle == 1
    remove_first = (
        input(
            "\nDo you want to remove the first (potentially abnormal) cycle i.e. Cycle == 1? (yes/no): "
        )
        .strip()
        .lower()
    )
    if remove_first == "yes":
        if 1.0 in df_interpolated["Cycle"].unique():
            print("Removing Cycle == 1...")
            df_interpolated = df_interpolated[df_interpolated["Cycle"] != 1].copy()

            # Renumber remaining cycles starting from 0
            unique_cycles = sorted(df_interpolated["Cycle"].unique())
            cycle_map = {old: new + 1 for new, old in enumerate(unique_cycles)}
            df_interpolated["Cycle"] = df_interpolated["Cycle"].map(cycle_map)
            print("Cycles renumbered starting from 1.")
        else:
            print("Cycle == 1 not found — nothing removed.")
    else:
        print("Keeping all cycles.")

    write = (
        input(
            "\nEnter 'yes' to save the interpolated file, or anything else to cancel: "
        )
        .strip()
        .lower()
    )
    if write == "yes":
        out_file = f"interp_{file}"
        df_interpolated.to_csv(out_file, index=False)
        print(f"✅ Interpolated data written to: {out_file}")
    else:
        print("Exiting without writing.")


if __name__ == "__main__":
    main()
