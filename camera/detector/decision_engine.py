def choose_response(risk):

    if risk == "HIGH":
        return "ACTIVATE_DETERRENT"

    elif risk == "MEDIUM":
        return "MONITOR"

    else:
        return "NO_ACTION"