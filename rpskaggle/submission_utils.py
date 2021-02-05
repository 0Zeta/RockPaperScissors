def generate_submissions():
    """
    Generates a single .py file for each agent
    """
    commons = [
        "agents/rpscontest_bots.py",
        "helpers.py",
        "policies.py",
        "agents/geometry_agent.py",
        "agents/iocaine_powder_policy.py",
        "agents/greenberg_policy.py",
        "agents/seed_searcher.py",
        "agents/anti_geo.py",
    ]
    agents = {
        "statistical_policy_ensemble": "agents/statistical_policy_ensemble_agent.py",
        "multi_armed_bandit": "agents/multi_armed_bandit.py",
        "neural_policy_ensemble": "agents/neural_policy_ensemble.py",
    }
    exclude_imports = ["agents/rpscontest_bots.py"]

    for agent in agents.keys():
        imports = set()
        other_lines = []
        for path in commons + [agents[agent]]:
            copy_imports = path not in exclude_imports
            with open(path, "r") as file:
                lines = file.readlines()
                for line in lines:
                    if copy_imports and "import" in line:
                        if "rpskaggle" not in line:
                            imports.add(line.strip() + "\n")
                    else:
                        other_lines.append(line)
        with open(agent + "_submission.py", "w") as file:
            file.writelines(["%%writefile " + agent + ".py\n"])
            file.writelines(imports)
            file.writelines(other_lines)


if __name__ == "__main__":
    generate_submissions()
