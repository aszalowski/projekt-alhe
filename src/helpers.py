from Parameters import UniformParameter, DiscreteParameter

def initializeXGBoostParameters():
    parameterList = []

    parameterList.append(UniformParameter(name="eta", bounds=(0, 1), mutationRate=0.5))
    parameterList.append(UniformParameter(name="gamma", bounds=(0, 10), mutationRate=1))
    parameterList.append(DiscreteParameter(name="max_depth", bounds=(1, 25), mutationRate=3))
    parameterList.append(UniformParameter(name="min_child_weight", bounds=(0, 10), mutationRate=3))
    parameterList.append(UniformParameter(name="max_delta_step", bounds=(0, 5), mutationRate=2))
    parameterList.append(UniformParameter(name="subsample", bounds=(0, 1), mutationRate=0.5))
    parameterList.append(UniformParameter(name="scale_pos_weight", bounds=(1, 10), mutationRate=2))

    return parameterList


if __name__== '__main__':
    # Quick test
    print(initializeXGBoostParameters())





