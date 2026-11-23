from pathlib import Path
import pandas as pd
import streamlit as st
import random

st.title("TV Scheduling using Genetic Algorithm")

file_path = Path(__file__).parent / "program_ratings.csv"
original_df = pd.read_csv(file_path)

st.subheader("Original Program Ratings")
st.dataframe(original_df)

st.write("Number of programs:", len(original_df))
st.write("Number of time slots:", len(original_df.columns) - 1)

modified_path = Path(__file__).parent / "program_ratings_modified.csv"

# Create a copy only if the modified file does not exist
if not modified_path.exists():
    original_df.to_csv(modified_path, index=False)

modified_df = pd.read_csv(modified_path)

st.subheader("Edit Modified Program Ratings")

edited_df = st.data_editor(
    modified_df,
    disabled=["Type of Program"],
    key="ratings_editor"
)

if st.button("Save Rating Changes"):
    edited_df.to_csv(modified_path, index=False)
    st.success("Modified ratings saved!")


    # Convert the modified ratings into a dictionary
rating_table = edited_df.set_index("Type of Program")

ratings = {
    program: row.tolist()
    for program, row in rating_table.iterrows()
}

all_programs = list(ratings.keys())
time_slots = list(rating_table.columns)


# Calculate the total rating of a schedule
def fitness_function(schedule):
    total_rating = 0

    for slot_index, program in enumerate(schedule):
        total_rating += ratings[program][slot_index]

    return total_rating


# Temporary example to check our fitness calculation
# Repeat the program list to fill all 18 hourly slots
example_schedule = (
    all_programs * len(time_slots)
)[:len(time_slots)]

example_table = pd.DataFrame({
    "Hour": time_slots,
    "Program": example_schedule,
    "Rating": [
        ratings[program][i]
        for i, program in enumerate(example_schedule)
    ]
})

st.subheader("Example Schedule — Fitness Check")
st.dataframe(example_table)

st.write(
    "Total rating:",
    round(fitness_function(example_schedule), 2)
)

# Create random schedules for the initial population
def initialize_population(population_size):
    population = []

    for _ in range(population_size):
        schedule = [
            random.choice(all_programs)
            for _ in time_slots
        ]
        population.append(schedule)

    return population


if st.button("Generate Initial Population"):
    population = initialize_population(50)

    population_summary = pd.DataFrame({
        "Schedule": range(1, len(population) + 1),
        "Total Rating": [
            fitness_function(schedule)
            for schedule in population
        ]
    })

    st.subheader("Initial Population")
    st.dataframe(population_summary)

    st.write("Number of schedules:", len(population))
    st.write("Hourly slots per schedule:", len(population[0]))


    # Selection: choose the best of 3 randomly sampled schedules
def select_parent(population):
    candidates = random.sample(population, 3)
    return max(candidates, key=fitness_function)


# Crossover: combine parts of two parent schedules
def crossover(parent1, parent2, crossover_rate):
    if random.random() < crossover_rate:
        point = random.randint(1, len(parent1) - 1)

        child1 = parent1[:point] + parent2[point:]
        child2 = parent2[:point] + parent1[point:]

        return child1, child2

    # If crossover does not happen, copy the parents
    return parent1.copy(), parent2.copy()

# Mutation: possibly replace one program in a schedule
def mutate(schedule, mutation_rate):
    mutated_schedule = schedule.copy()

    if random.random() < mutation_rate:
        position = random.randrange(len(mutated_schedule))
        current_program = mutated_schedule[position]

        alternatives = [
            program
            for program in all_programs
            if program != current_program
        ]

        mutated_schedule[position] = random.choice(alternatives)

    return mutated_schedule

def genetic_algorithm(
    crossover_rate,
    mutation_rate,
    generations=100,
    population_size=50,
    elitism_size=2
):
    population = initialize_population(population_size)

    # Record the best fitness before evolution starts
    history = [max(fitness_function(s) for s in population)]

    for generation in range(generations):
        # Rank schedules from highest to lowest rating
        population.sort(key=fitness_function, reverse=True)

        # Preserve the best schedules unchanged
        new_population = [
            schedule.copy()
            for schedule in population[:elitism_size]
        ]

        while len(new_population) < population_size:
            parent1 = select_parent(population)
            parent2 = select_parent(population)

            child1, child2 = crossover(
                parent1, parent2, crossover_rate
            )

            child1 = mutate(child1, mutation_rate)
            child2 = mutate(child2, mutation_rate)

            new_population.append(child1)

            if len(new_population) < population_size:
                new_population.append(child2)

        population = new_population

        history.append(
            max(fitness_function(s) for s in population)
        )

    best_schedule = max(population, key=fitness_function)

    return best_schedule, history


st.subheader("Genetic Algorithm Parameters")

crossover_rate = st.slider(
    "Crossover Rate (CO_R)",
    min_value=0.0,
    max_value=0.95,
    value=0.8,
    step=0.05
)

mutation_rate = st.slider(
    "Mutation Rate (MUT_R)",
    min_value=0.01,
    max_value=0.05,
    value=0.05,
    step=0.01
)

if "trials" not in st.session_state:
    st.session_state.trials = []

if st.button("Run Genetic Algorithm"):
    best_schedule, history = genetic_algorithm(
        crossover_rate=crossover_rate,
        mutation_rate=mutation_rate
    )

    st.session_state.trials.append({
        "Trial": len(st.session_state.trials) + 1,
        "CO_R": crossover_rate,
        "MUT_R": mutation_rate,
        "Total Rating": fitness_function(best_schedule),
        "Schedule": best_schedule.copy(),
        "History": history.copy()
    })


    result_table = pd.DataFrame({
        "Hour": time_slots,
        "Program": best_schedule,
        "Rating": [
            ratings[program][i]
            for i, program in enumerate(best_schedule)
        ]
    })

    st.subheader("Best Schedule Found")
    st.dataframe(result_table)

    st.metric(
        "Total Rating",
        f"{fitness_function(best_schedule):.2f}"
    )

    st.subheader("Best Rating Across Generations")
    st.line_chart(
        pd.DataFrame({
            "Generation": range(len(history)),
            "Best Rating": history
        }).set_index("Generation")
    )

st.subheader("Saved Trial Results")

if st.session_state.trials:
    summary = pd.DataFrame([
        {
            "Trial": trial["Trial"],
            "CO_R": trial["CO_R"],
            "MUT_R": trial["MUT_R"],
            "Total Rating": round(trial["Total Rating"], 2)
        }
        for trial in st.session_state.trials
    ])

    st.dataframe(summary)

    for trial in st.session_state.trials:
        with st.expander(f"Trial {trial['Trial']} Schedule"):
            schedule = trial["Schedule"]

            st.dataframe(pd.DataFrame({
                "Hour": time_slots,
                "Program": schedule,
                "Rating": [
                    ratings[program][i]
                    for i, program in enumerate(schedule)
                ]
            }))

    st.download_button(
        "Download Trial Summary",
        data=summary.to_csv(index=False).encode("utf-8"),
        file_name="trial_summary.csv",
        mime="text/csv"
    )

    if st.button("Clear Saved Trials"):
        st.session_state.trials = []
        st.rerun()