import streamlit as st
import os
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

# CrewAI libraries (adapt based on your environment/project)
from crewai import Crew, Agent, Task, Process
from langchain_openai import ChatOpenAI

# ------------------------------------------------------------------
# 1) CONFIGURATION & OPENAI INIT
# ------------------------------------------------------------------
openai_api_key = st.secrets["OPENAI_API_KEY"]  # Ensure you have this key in Streamlit secrets
os.environ["OPENAI_API_KEY"] = openai_api_key

# Initialize OpenAI LLM
openai_llm = ChatOpenAI(
    model_name="gpt-4o-mini",  # Use "gpt-4" or another available model
    temperature=0.7
)

# ------------------------------------------------------------------
# 1b) DICTIONARY FOR NATION -> CITY -> ZONES
# ------------------------------------------------------------------
city_zones = {
    "Italy": {
        "Milan": ["Historic Center", "Navigli", "Porta Nuova", "Città Studi", "Isola", "Brera", "Porta Romana"],
        "Rome": ["Trastevere", "Monti", "Prati", "Historic Center", "Parioli", "San Giovanni", "Eur"],
        "Turin": ["Center", "San Salvario", "Crocetta", "Cit Turin", "Vanchiglia", "Lingotto"],
        "Naples": ["Vomero", "Chiaia", "Historic Center", "Posillipo", "Fuorigrotta", "Arenella"],
        "Florence": ["Duomo", "Oltrarno", "Santa Croce", "San Lorenzo", "Campo di Marte", "Isolotto"],
        "Bologna": ["Historic Center", "Bolognina", "Saragozza", "Murri", "Mazzini", "San Donato"],
        "Venice": ["San Marco", "Dorsoduro", "Cannaregio", "Castello", "Santa Croce", "Giudecca"]
    },
    "Spain": {
        "Madrid": ["Salamanca", "Chueca", "Malasaña", "Center", "Retiro", "Argüelles", "La Latina"],
        "Barcelona": ["Eixample", "Gràcia", "Barceloneta", "Gòtic", "El Born", "Poble-sec", "Sants"],
        "Valencia": ["Ciutat Vella", "El Carmen", "Ruzafa", "Patraix", "Benimaclet", "La Malvarrosa"],
        "Seville": ["Triana", "Santa Cruz", "Nervión", "Macarena", "Alameda", "Los Remedios"]
    },
    "France": {
        "Paris": ["Le Marais", "Saint-Germain", "Montmartre", "Bastille", "La Défense", "Champs-Élysées"],
        "Marseille": ["Vieux Port", "Le Panier", "La Joliette", "Noailles", "Endoume", "Le Rouet"],
        "Lyon": ["Presqu'île", "Vieux Lyon", "Part-Dieu", "Croix-Rousse", "Confluence"],
        "Nice": ["Vieux Nice", "Port", "Promenade des Anglais", "Libération", "Cimiez"]
    },
    "England": {
        "London": ["Mayfair", "Soho", "Camden", "Chelsea", "Kensington", "Notting Hill", "Shoreditch"],
        "Manchester": ["Northern Quarter", "Deansgate", "Salford Quays", "Ancoats", "Castlefield"],
        "Birmingham": ["Jewellery Quarter", "Digbeth", "Edgbaston", "Moseley", "Harborne"],
        "Liverpool": ["Albert Dock", "The Baltic Triangle", "Anfield", "Wavertree", "Toxteth"]
    },
    "Germany": {
        "Berlin": ["Mitte", "Prenzlauer Berg", "Kreuzberg", "Friedrichshain", "Charlottenburg"],
        "Munich": ["Old Town", "Schwabing", "Maxvorstadt", "Glockenbachviertel", "Haidhausen"],
        "Hamburg": ["Sankt Pauli", "Altona", "HafenCity", "Eimsbüttel", "Blankenese"],
        "Frankfurt": ["Innenstadt", "Sachsenhausen", "Nordend", "Bornheim", "Bockenheim"]
    },
    "Netherlands": {
        "Amsterdam": ["Jordaan", "De Pijp", "Nieuw-West", "Zuid", "Oost", "Centrum"],
        "Rotterdam": ["Kop van Zuid", "Centrum", "Blijdorp", "Oude Noorden", "Delfshaven"],
        "The Hague": ["Centrum", "Scheveningen", "Loosduinen", "Escamp", "Bezuidenhout"],
        "Utrecht": ["Binnenstad", "Leidsche Rijn", "Oost", "Zuilen", "Lombok"]
    }
}

# ------------------------------------------------------------------
# 2) DEFINE SKILL-BASED AGENTS (EXISTING)
# ------------------------------------------------------------------

Pricing_Expert = Agent(
    role="Pricing Expert",
    goal="Analyze recent comparable sales, local market data, and property features to propose an optimal price.",
    backstory=(
        """Expert in real estate pricing:
        - Skilled in evaluating property comps
        - Understands supply & demand factors
        - Provides data-driven listing or offer prices
        """
    ),
    llm=openai_llm,
    verbose=False,
    allow_delegation=False,
    max_iter=3,
    memory=True,
)

Negotiation_Expert = Agent(
    role="Negotiation Expert",
    goal="Offer negotiation strategies appropriate for either buyers or sellers.",
    backstory=(
        """Expert in real estate negotiations:
        - Skilled in buyer/seller tactics
        - Aware of psychological and market leverage
        - Offers practical, step-by-step negotiation advice
        """
    ),
    llm=openai_llm,
    verbose=False,
    allow_delegation=False,
    max_iter=3,
    memory=True,
)

Mortgage_Expert = Agent(
    role="Mortgage Expert",
    goal="Propose feasible financing options, assess interest rates, and explain pros/cons of different mortgage structures.",
    backstory=(
        """Expert in mortgage financing:
        - Knowledgeable about fixed/variable rates
        - Understands risk management and regulations
        - Explains monthly payment structures
        """
    ),
    llm=openai_llm,
    verbose=False,
    allow_delegation=False,
    max_iter=3,
    memory=True,
)

Market_Analysis_Expert = Agent(
    role="Market Analysis Expert",
    goal="Forecast local real estate prices, analyzing supply/demand, macroeconomic trends, and relevant policies.",
    backstory=(
        """Expert in market forecasting:
        - Skilled in reading economic indicators
        - Can provide short/medium term outlook
        - Highlights risks and opportunities
        """
    ),
    llm=openai_llm,
    verbose=False,
    allow_delegation=False,
    max_iter=3,
    memory=True,
)

Legal_Expert = Agent(
    role="Legal Expert",
    goal="Outline legal documents, local regulations, and potential pitfalls for real estate transactions.",
    backstory=(
        """Expert in real estate legality:
        - Knows about contract requirements
        - Understands property disclosure rules
        - Familiar with local regulatory frameworks
        """
    ),
    llm=openai_llm,
    verbose=False,
    allow_delegation=False,
    max_iter=3,
    memory=True,
)

# ------------------------------------------------------------------
# 2b) DEFINE NEW AGENTS
# ------------------------------------------------------------------

Investment_Analyst = Agent(
    role="Investment Analyst",
    goal="Analyze the property from a long-term investment perspective.",
    backstory=(
        """Expert in real estate investments:
        - Calculates ROI, analyzes long-term trends
        - Estimates appreciation, forecasts risks and potential gains
        - Advises strategies to maximize returns over time
        """
    ),
    llm=openai_llm,
    verbose=False,
    allow_delegation=False,
    max_iter=3,
    memory=True,
)

Rental_Market_Expert = Agent(
    role="Rental Market Expert",
    goal="Analyze the local rental market and provide strategies for short-term and long-term rentals.",
    backstory=(
        """Expert in the rental market:
        - Knowledgeable about average rental prices (residential, Airbnb)
        - Evaluates supply/demand in different zones
        - Suggests strategies to maximize rental income
        """
    ),
    llm=openai_llm,
    verbose=False,
    allow_delegation=False,
    max_iter=3,
    memory=True,
)

Sustainability_Expert = Agent(
    role="Sustainability Expert",
    goal="Provide eco-friendly suggestions to increase market value and reduce management costs.",
    backstory=(
        """Expert in green solutions:
        - Recommends structural improvements (insulation, solar panels)
        - Evaluates economic and environmental benefits
        - Estimates potential savings and impact on resale value
        """
    ),
    llm=openai_llm,
    verbose=False,
    allow_delegation=False,
    max_iter=3,
    memory=True,
)

# ------------------------------------------------------------------
# 3) DEFINE TASK FUNCTIONS (EXISTING)
# ------------------------------------------------------------------

def pricing_task(location, property_type, budget):
    prompt = (
        f"Property Type: {property_type}\n"
        f"Location: {location}\n"
        f"Budget or Valuation: {budget} EUR\n\n"
        "Propose a fair listing/offer price range based on market comps, recent transactions, and supply/demand."
    )
    return Task(
        description=prompt,
        expected_output="A concise analysis and recommended price range, with short justification.",
        agent=Pricing_Expert
    )

def negotiation_task(location, user_role):
    prompt = (
        f"User Role: {user_role}\n"
        f"Location: {location}\n\n"
        "Provide negotiation strategies appropriate for this role, considering current market conditions."
    )
    return Task(
        description=prompt,
        expected_output="Specific, actionable negotiation advice targeted to the role.",
        agent=Negotiation_Expert
    )

def mortgage_task(location, budget, interest_rate, mortgage_years):
    prompt = (
        f"Location: {location}\n"
        f"Budget: {budget} EUR\n"
        f"Interest Rate: {interest_rate}%\n"
        f"Mortgage Duration: {mortgage_years} years\n\n"
        "Recommend suitable mortgage structures (fixed, variable, etc.), monthly payment estimates, and key pros/cons."
    )
    return Task(
        description=prompt,
        expected_output="A brief list or paragraph detailing mortgage options and risk considerations.",
        agent=Mortgage_Expert
    )

def market_analysis_task(location, horizon, inflation_rate):
    prompt = (
        f"Location: {location}\n"
        f"Forecast Horizon: {horizon} months\n"
        f"Inflation Rate: {inflation_rate}%\n\n"
        "Analyze the real estate market conditions and forecast short-term price trends, highlighting potential risks/opportunities."
    )
    return Task(
        description=prompt,
        expected_output="A short overview of market trends, likely price direction, supply/demand, and macro factors.",
        agent=Market_Analysis_Expert
    )

def legal_task(location):
    prompt = (
        f"Location: {location}\n\n"
        "Outline the basic legal steps and documents involved in a real estate transaction here, "
        "including any special local requirements."
    )
    return Task(
        description=prompt,
        expected_output="A concise list of important documents, steps, and pitfalls in local property law.",
        agent=Legal_Expert
    )

# ------------------------------------------------------------------
# 3b) DEFINE NEW TASK FUNCTIONS
# ------------------------------------------------------------------

def investment_task(location, property_type, budget, property_condition, surface, num_rooms, num_bathrooms):
    prompt = (
        f"Location: {location}\n"
        f"Property Type: {property_type}\n"
        f"Budget / Valuation: {budget} EUR\n"
        f"Condition: {property_condition}\n"
        f"Surface: {surface} sqm\n"
        f"Rooms: {num_rooms}, Bathrooms: {num_bathrooms}\n\n"
        "Analyze this property from a long-term investment perspective, estimating ROI, appreciation forecasts, and key risks."
    )
    return Task(
        description=prompt,
        expected_output="ROI analysis, risks, and appreciation estimates for a long-term investment.",
        agent=Investment_Analyst
    )

def rental_task(location, property_type, surface, num_rooms, num_bathrooms, short_term_interest):
    prompt = (
        f"Location: {location}\n"
        f"Property Type: {property_type}\n"
        f"Surface: {surface} sqm\n"
        f"Rooms: {num_rooms}, Bathrooms: {num_bathrooms}\n"
        f"Short-term interest: {short_term_interest}\n\n"
        "Provide an analysis of the rental market (average prices, demand) and strategies to maximize income, including short-term rentals."
    )
    return Task(
        description=prompt,
        expected_output="Analysis of rental prices, supply/demand, and rental strategy suggestions.",
        agent=Rental_Market_Expert
    )

def sustainability_task(location, property_condition, budget):
    prompt = (
        f"Location: {location}\n"
        f"Property Condition: {property_condition}\n"
        f"Budget: {budget} EUR\n\n"
        "Recommend eco-friendly improvements (solar panels, insulation, etc.) to increase value and reduce management costs. Also estimate potential savings."
    )
    return Task(
        description=prompt,
        expected_output="List of sustainable recommendations, approximate costs, and medium-long term savings.",
        agent=Sustainability_Expert
    )

# ------------------------------------------------------------------
# 4) OPTIONAL HELPER FUNCTIONS FOR CHARTS (EXISTING)
# ------------------------------------------------------------------

def show_pricing_chart(budget):
    """
    Mock example: Create fictitious 'comparable' properties around the user's budget.
    """
    st.subheader("Pricing Analysis Chart: Comparable Properties")

    np.random.seed(42)  # For reproducibility
    num_comps = 5
    comps_ids = [f"Comp {i+1}" for i in range(num_comps)]
    comps_prices = [budget * random.uniform(0.8, 1.2) for _ in range(num_comps)]

    df_comps = pd.DataFrame({
        "Property": comps_ids,
        "Price": comps_prices
    })

    recommended_price = budget

    fig, ax = plt.subplots()
    ax.bar(df_comps["Property"], df_comps["Price"], color="gray", label="Comparable Props")
    ax.axhline(y=recommended_price, color="red", linestyle="--", label="Recommended Price")
    ax.set_xlabel("Comparable Properties")
    ax.set_ylabel("Price (EUR)")
    ax.set_title("Comparable Prices vs. Recommended Listing/Offer")
    ax.legend()
    st.pyplot(fig)

def show_negotiation_chart():
    st.subheader("Negotiation Tactics: Risk/Reward Overview")

    tactics = ["Aggressive Offer", "Low-Ball", "Flexible Closing", "Escalation Clause", "All-Cash Offer"]
    risk_scores = [random.uniform(1, 5) for _ in tactics]
    reward_scores = [random.uniform(1, 5) for _ in tactics]

    df_neg = pd.DataFrame({
        "Tactic": tactics,
        "Risk": risk_scores,
        "Reward": reward_scores
    })

    fig, ax = plt.subplots()
    x = np.arange(len(tactics))
    width = 0.35

    ax.bar(x - width/2, df_neg["Risk"], width, label="Risk", color="#F5B041")
    ax.bar(x + width/2, df_neg["Reward"], width, label="Reward", color="#58D68D")

    ax.set_ylabel("Score (1–5)")
    ax.set_title("Negotiation Tactic Risk vs. Reward")
    ax.set_xticks(x)
    ax.set_xticklabels(tactics, rotation=45, ha="right")
    ax.legend()
    st.pyplot(fig)

def show_mortgage_chart():
    st.subheader("Mortgage Payment Comparison")

    mortgage_scenarios = {
        "30-yr Fixed at 3.0%": random.randint(800, 1200),
        "15-yr Fixed at 2.8%": random.randint(1400, 1900),
        "Variable at ~2.5%":  random.randint(700, 1300)
    }

    fig, ax = plt.subplots()
    ax.bar(mortgage_scenarios.keys(), mortgage_scenarios.values(), color=["#2E86C1", "#58D68D", "#F5B041"])
    ax.set_title("Monthly Payment Comparison")
    ax.set_ylabel("Monthly Payment (EUR)")

    for i, (k, v) in enumerate(mortgage_scenarios.items()):
        ax.text(i, v + 20, f"{v} €", ha='center', fontweight='bold')
    st.pyplot(fig)

def show_market_chart(location, simulation_horizon, inflation_rate):
    st.subheader("Market Forecast Chart")

    months = list(range(1, simulation_horizon + 1))
    monthly_growth_rate = 0.05 / 12  # Example fictitious
    forecast = []
    lower_bound = []
    upper_bound = []
    current_price = 3000  # Example baseline (€/sqm)

    for _ in months:
        current_price *= (1 + monthly_growth_rate)
        inflation_factor = 1 + (inflation_rate / 100) / 12
        current_price *= inflation_factor

        low = current_price * random.uniform(0.90, 0.95)
        high = current_price * random.uniform(1.05, 1.10)

        forecast.append(current_price)
        lower_bound.append(low)
        upper_bound.append(high)

    df_forecast = pd.DataFrame({
        "Month": months,
        "Forecast": forecast,
        "Lower": lower_bound,
        "Upper": upper_bound
    })

    fig, ax = plt.subplots()
    ax.plot(df_forecast["Month"], df_forecast["Forecast"], color="#0C4B8E", label="Forecast")
    ax.fill_between(
        df_forecast["Month"],
        df_forecast["Lower"],
        df_forecast["Upper"],
        color="#0C4B8E",
        alpha=0.2,
        label="Confidence Interval"
    )
    ax.set_title(f"{location} - Price Forecast (Next {simulation_horizon} Months)")
    ax.set_xlabel("Month")
    ax.set_ylabel("Price per sqm (€)")
    ax.legend()
    st.pyplot(fig)

def show_legal_flowchart():
    st.subheader("Legal Process Flow")
    steps = [
        "1. Offer & Acceptance",
        "2. Contract Drafting & Signing",
        "3. Disclosure & Inspection",
        "4. Mortgage Approval (if needed)",
        "5. Title Check & Insurance",
        "6. Closing & Transfer of Ownership"
    ]
    st.markdown("**Typical Real Estate Transaction Steps:**")
    for step in steps:
        st.markdown(f"- {step}")

# ------------------------------------------------------------------
# 5) STREAMLIT APP
# ------------------------------------------------------------------
def main():
    st.title("Modular Real Estate Advisor (Skill-Based Agents)")
    st.header("Welcome to Your Real Estate Advisor")

    # Initialize session state variables
    if 'user_role' not in st.session_state:
        st.session_state.user_role = None

    # Function to reset the app to the initial state
    def reset():
        st.session_state.user_role = None

    # Screen 1: Role Selection
    if st.session_state.user_role is None:
        st.subheader("Select Your Profile")

        user_role = st.radio(
            "Who are you?",
            ("Buyer", "Seller", "Real Estate Agency")
        )

        if st.button("Continue"):
            if user_role in ["Buyer", "Seller", "Real Estate Agency"]:
                st.session_state.user_role = user_role

    # Screen 2: Dynamic Form based on Role
    else:
        st.subheader(f"{st.session_state.user_role} Details")

        # Localization Section
        st.markdown("### Localization")
        nation = st.selectbox("Select Nation", list(city_zones.keys()))
        city = st.selectbox("Select City", list(city_zones[nation].keys()))
        zones = st.multiselect("Select one or more Zones (optional)", city_zones[nation][city])

        # Create a location string that includes nation, city, and zones
        location_str = f"{city}, {nation}"
        if zones:
            location_str += f" | Zones: {', '.join(zones)}"

        # Form based on user role
        with st.form("real_estate_form"):
            st.markdown("### Enter Property Details")

            if st.session_state.user_role == "Buyer":
                st.markdown("#### Buyer Information")
                budget = st.slider("Budget (€)", 50000, 2000000, 300000, 5000)
                interest_rate = st.slider("Mortgage Interest Rate (%)", 0.0, 10.0, 3.0, 0.1)
                mortgage_years = st.selectbox("Mortgage Duration (years)", [10, 15, 20, 25, 30])
                property_type = st.selectbox("Property Type", ["One-Bedroom", "Two-Bedroom", "Villa", "Loft"])
                parking = st.selectbox("Parking Availability", ["None", "Box", "Car Space"])

            elif st.session_state.user_role == "Seller":
                st.markdown("#### Seller Information")
                valuation = st.slider("Estimated Property Valuation (€)", 50000, 2000000, 400000, 5000)
                property_condition = st.selectbox("Property Condition", ["New", "Renovated", "Needs Renovation"])
                selling_priority = st.radio("Main Objective", ["Quick Sale", "Maximize Profit"])
                surface = st.number_input("Surface Area (sqm)", min_value=20, max_value=500, value=80, step=5)
                num_rooms = st.number_input("Number of Rooms", min_value=1, max_value=10, value=3, step=1)
                property_type = st.selectbox("Property Type", ["One-Bedroom", "Two-Bedroom", "Villa", "Loft"])  # Added property_type

            elif st.session_state.user_role == "Real Estate Agency":
                st.markdown("#### Agency Information")
                num_properties = st.slider("Number of Properties Managed", 1, 100, 10)
                project_type = st.selectbox("Project Type", ["Investment", "Short-Term Rental", "Renovation"])
                forecast_horizon = st.slider("Forecast Horizon (months)", 6, 36, 12)
                property_type = st.selectbox("Property Type", ["One-Bedroom", "Two-Bedroom", "Villa", "Loft"])  # Added property_type

            # Common Preferences/Constraints
            st.markdown("### Preferences & Constraints")
            inflation_rate = st.slider("Expected Inflation Rate (%)", 0.0, 10.0, 2.5, 0.1)
            simulation_horizon = st.selectbox("Simulation Horizon (months)", [6, 12, 24, 36])

            submitted = st.form_submit_button("Generate Advice & Charts")

        if submitted:
            st.write("**Running the relevant skill-based agents...**")

            # Display selected location
            st.write(f"**Selected Location:** {location_str}")

            # Decide which tasks to launch based on role and objective
            tasks = []

            user_goal = ""
            short_term_interest = False

            if st.session_state.user_role == "Buyer":
                user_goal = "Buy"
                short_term_interest = False  # Adjust as needed

            elif st.session_state.user_role == "Seller":
                user_goal = selling_priority

            elif st.session_state.user_role == "Real Estate Agency":
                user_goal = project_type

            # Assign tasks based on user_role and user_goal
            # Buyer
            if st.session_state.user_role == "Buyer":
                task_neg = negotiation_task(location_str, st.session_state.user_role)
                task_mort = mortgage_task(location_str, budget, interest_rate, mortgage_years)
                task_ma = market_analysis_task(location_str, simulation_horizon, inflation_rate)
                task_legal_ = legal_task(location_str)
                tasks.extend([task_neg, task_mort, task_ma, task_legal_])

            # Seller
            elif st.session_state.user_role == "Seller":
                task_pr = pricing_task(location_str, property_type, valuation)
                task_neg = negotiation_task(location_str, st.session_state.user_role)
                task_ma = market_analysis_task(location_str, simulation_horizon, inflation_rate)
                task_legal_ = legal_task(location_str)
                tasks.extend([task_pr, task_neg, task_ma, task_legal_])

            # Real Estate Agency
            elif st.session_state.user_role == "Real Estate Agency":
                task_pr = pricing_task(location_str, property_type, num_properties * 100000)  # Example: total budget = num_properties * average
                task_neg = negotiation_task(location_str, st.session_state.user_role)
                task_ma = market_analysis_task(location_str, simulation_horizon, inflation_rate)
                task_legal_ = legal_task(location_str)
                tasks.extend([task_pr, task_neg, task_ma, task_legal_])

            # Additional tasks based on user_goal
            if st.session_state.user_role == "Seller":
                if user_goal == "Maximize Profit":
                    task_inv = investment_task(location_str, property_type, valuation, property_condition, surface, num_rooms, num_bathrooms=1)
                    tasks.append(task_inv)
                elif user_goal == "Quick Sale":
                    # Quick Sale might require different strategies
                    pass  # Add specific tasks if needed

            # Always add sustainability solutions
            if st.session_state.user_role == "Buyer":
                task_sust = sustainability_task(location_str, property_condition="New", budget=budget)
            elif st.session_state.user_role == "Seller":
                task_sust = sustainability_task(location_str, property_condition=property_condition, budget=valuation)
            elif st.session_state.user_role == "Real Estate Agency":
                task_sust = sustainability_task(location_str, property_condition="Renovated", budget=num_properties * 100000)  # Example
            tasks.append(task_sust)

            # Create a Crew with ALL agents (existing + new)
            crew = Crew(
                agents=[
                    Pricing_Expert,
                    Negotiation_Expert,
                    Mortgage_Expert,
                    Market_Analysis_Expert,
                    Legal_Expert,
                    Investment_Analyst,
                    Rental_Market_Expert,
                    Sustainability_Expert
                ],
                tasks=tasks,
                process=Process.sequential,  # or Process.parallel
                full_output=True
            )
            crew.kickoff()

            # Retrieve the results
            results = {
                "Pricing Expert": "",
                "Negotiation Expert": "",
                "Mortgage Expert": "",
                "Market Analysis Expert": "",
                "Legal Expert": "",
                "Investment Analyst": "",
                "Rental Market Expert": "",
                "Sustainability Expert": ""
            }

            for task in tasks:
                agent_role = task.agent.role
                if agent_role in results:
                    results[agent_role] = task.output.raw

            # Display the results
            # Pricing
            if results["Pricing Expert"]:
                st.header("Pricing Expert's Advice")
                st.write(results["Pricing Expert"])
                budget_for_chart = valuation if st.session_state.user_role == "Seller" else budget
                st.markdown("---")
                show_pricing_chart(budget=budget_for_chart)

            # Negotiation
            if results["Negotiation Expert"]:
                st.header("Negotiation Expert's Advice")
                st.write(results["Negotiation Expert"])
                st.markdown("---")
                show_negotiation_chart()

            # Mortgage
            if results["Mortgage Expert"]:
                st.header("Mortgage Expert's Advice")
                st.write(results["Mortgage Expert"])
                st.markdown("---")
                show_mortgage_chart()

            # Market Analysis
            if results["Market Analysis Expert"]:
                st.header("Market Analysis Expert's Advice")
                st.write(results["Market Analysis Expert"])
                st.markdown("---")
                show_market_chart(location_str, simulation_horizon, inflation_rate)

            # Legal
            if results["Legal Expert"]:
                st.header("Legal Expert's Advice")
                st.write(results["Legal Expert"])
                st.markdown("---")
                show_legal_flowchart()

            # Investment
            if results["Investment Analyst"]:
                st.header("Investment Analyst's Advice")
                st.write(results["Investment Analyst"])
                st.markdown("---")
                # Example: You could add a custom graph, e.g., ROI vs. Time

            # Rental
            if results["Rental Market Expert"]:
                st.header("Rental Market Expert's Advice")
                st.write(results["Rental Market Expert"])
                st.markdown("---")
                # Example: You could add a graph on average rental prices or occupancy rates

            # Sustainability
            if results["Sustainability Expert"]:
                st.header("Sustainability Expert's Advice")
                st.write(results["Sustainability Expert"])
                st.markdown("---")
                # Example: You could add a graph with potential savings or installation costs

            st.success("**All advice and charts generated!**")

        # Back Button
        if st.session_state.user_role is not None:
            if st.button("🔙 Go Back"):
                reset()

if __name__ == "__main__":
    main()
