
def get_params_combinations(grid_params):
    keys = list(grid_params.keys())
    comb_params = [{keys[0]: x} for x in grid_params[keys[0]]]
    for i in range(1, len(keys)):
        comb_params = [{**x, **{keys[i]: y}} for x in comb_params for y in grid_params[keys[i]]]
    return comb_params
