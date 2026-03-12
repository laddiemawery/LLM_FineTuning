"""Specialized parser for converting structured training log data into natural language."""

from datetime import datetime

import pandas as pd


class TrainingLogParser:
    """Convert structured exercise/training data into natural language narratives."""

    # Common column name mappings (normalize various naming conventions)
    COLUMN_ALIASES = {
        "date": ["date", "workout_date", "session_date", "day", "timestamp"],
        "exercise": ["exercise", "exercise_name", "movement", "lift", "activity"],
        "sets": ["sets", "num_sets", "set_count"],
        "reps": ["reps", "repetitions", "num_reps", "rep_count"],
        "weight": ["weight", "load", "resistance", "lbs", "kg", "weight_lbs", "weight_kg"],
        "rpe": ["rpe", "intensity", "effort", "rir"],
        "duration": ["duration", "time", "minutes", "seconds", "duration_min"],
        "distance": ["distance", "miles", "km", "meters"],
        "notes": ["notes", "comments", "remarks", "coach_notes"],
        "bodyweight": ["bodyweight", "bw", "body_weight"],
        "rest": ["rest", "rest_time", "rest_seconds", "rest_period"],
    }

    @classmethod
    def normalize_columns(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize column names to standard names."""
        col_map = {}
        for standard, aliases in cls.COLUMN_ALIASES.items():
            for alias in aliases:
                if alias in df.columns.str.lower().tolist():
                    original = df.columns[df.columns.str.lower() == alias][0]
                    col_map[original] = standard
                    break
        return df.rename(columns=col_map)

    @classmethod
    def narrate_row(cls, row: dict) -> str:
        """Convert a single training log row into a natural language sentence."""
        parts = []

        # Date
        if "date" in row and pd.notna(row["date"]):
            try:
                dt = pd.to_datetime(row["date"])
                parts.append(f"On {dt.strftime('%B %d, %Y')}")
            except (ValueError, TypeError):
                parts.append(f"On {row['date']}")

        # Exercise
        exercise = row.get("exercise", "an exercise")
        if pd.notna(exercise):
            exercise = str(exercise).strip()
        else:
            exercise = "an exercise"

        # Sets x Reps x Weight
        sets = row.get("sets")
        reps = row.get("reps")
        weight = row.get("weight")

        if pd.notna(sets) and pd.notna(reps):
            rep_str = f"{int(sets)} sets of {int(reps)} reps of {exercise}"
            if pd.notna(weight):
                rep_str += f" at {weight} lbs"
            parts.append(f"the client performed {rep_str}")
        elif pd.notna(sets):
            parts.append(f"the client performed {int(sets)} sets of {exercise}")
        else:
            parts.append(f"the client performed {exercise}")

        # Duration / Distance (for cardio)
        duration = row.get("duration")
        distance = row.get("distance")
        if pd.notna(duration):
            parts.append(f"for {duration} minutes")
        if pd.notna(distance):
            parts.append(f"covering {distance}")

        # RPE
        rpe = row.get("rpe")
        if pd.notna(rpe):
            rpe_val = float(rpe)
            if rpe_val <= 5:
                intensity = "low difficulty"
            elif rpe_val <= 7:
                intensity = "moderate difficulty"
            elif rpe_val <= 8.5:
                intensity = "high difficulty"
            else:
                intensity = "near-maximal effort"
            parts.append(f"with an RPE of {rpe_val}, indicating {intensity}")

        # Rest
        rest = row.get("rest")
        if pd.notna(rest):
            parts.append(f"with {rest} seconds of rest between sets")

        # Notes
        notes = row.get("notes")
        if pd.notna(notes) and str(notes).strip():
            parts.append(f"(Note: {str(notes).strip()})")

        sentence = ", ".join(parts) + "."
        # Capitalize first letter
        return sentence[0].upper() + sentence[1:]

    @classmethod
    def narrate_session(cls, session_rows: list[dict], session_date: str = "") -> str:
        """Convert a full training session into a narrative paragraph."""
        if not session_rows:
            return ""

        lines = []
        if session_date:
            lines.append(f"Training session on {session_date}:")
        else:
            lines.append("Training session:")

        for row in session_rows:
            lines.append(f"  - {cls.narrate_row(row)}")

        # Add summary
        total_sets = sum(int(r.get("sets", 0)) for r in session_rows if pd.notna(r.get("sets")))
        exercises = [r.get("exercise", "") for r in session_rows if pd.notna(r.get("exercise"))]
        if total_sets and exercises:
            lines.append(
                f"\nSession summary: {len(exercises)} exercises, "
                f"{total_sets} total sets across {', '.join(str(e) for e in exercises)}."
            )

        return "\n".join(lines)

    @classmethod
    def dataframe_to_narratives(
        cls, df: pd.DataFrame, group_by_date: bool = True
    ) -> list[dict]:
        """Convert an entire DataFrame of training logs into narrated text sections."""
        df = cls.normalize_columns(df)
        sections = []

        if group_by_date and "date" in df.columns:
            for date_val, group in df.groupby("date"):
                rows = group.to_dict("records")
                try:
                    date_str = pd.to_datetime(date_val).strftime("%B %d, %Y")
                except (ValueError, TypeError):
                    date_str = str(date_val)
                narrative = cls.narrate_session(rows, date_str)
                sections.append({
                    "title": f"Training Session - {date_str}",
                    "text": narrative,
                    "metadata": {
                        "date": str(date_val),
                        "num_exercises": len(rows),
                        "content_type": "tabular",
                    },
                })
        else:
            # No date grouping - narrate each row individually
            for _, row in df.iterrows():
                narrative = cls.narrate_row(row.to_dict())
                sections.append({
                    "title": "Training Log Entry",
                    "text": narrative,
                    "metadata": {"content_type": "tabular"},
                })

        return sections
