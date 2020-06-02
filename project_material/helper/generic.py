def mean_r0(model):
    return model.mean_r0()

def return_time(model):
    return model._current_timer

def return_total_n(model):
    return model.num_nodes

def cumulative_total_test_done(model):
    return model.cumulative_test_done

def cumulative_total_infectious(model):
    return model.cumulative_infectious_cases

def cumulative_total_dead(model):
    return model.cumulative_dead_cases

def cumulative_total_infectious_test_confirmed(model):
    return model.cumulative_total_infectious_test_confirmed()

def cumulative_total_dead_test_confirmed(model):
    return model.cumulative_total_dead_test_confirmed()

def rate_cumulative_infectious(model):
    return model.rate_cumulative_infectious()

def rate_cumulative_dead(model):
    return model.rate_cumulative_dead()

def rate_cumulative_infectious_test_confirmed(model):
    return model.rate_cumulative_infectious_test_confirmed()

def rate_cumulative_dead_test_confirmed(model):
    return model.rate_cumulative_dead_test_confirmed()

def rate_cumulative_test_done(model):
    return model.rate_cumulative_test_done()