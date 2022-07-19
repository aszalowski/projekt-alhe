from Parameters import UniformParameter, DiscreteParameter

def initializeXGBoostParameters(variation='global', defaultParamValues=None):
    parameterList = []

    if defaultParamValues:
        assert isinstance(defaultParamValues, dict)
    else:
        defaultParamValues = dict()

    if variation == 'local':
        parameterList.append(UniformParameter(name="eta", bounds=(0, 1), mutationRate=0.4, defaultValue=defaultParamValues.get("eta", None)))
        parameterList.append(UniformParameter(name="gamma", bounds=(0, 10), mutationRate=0.4, defaultValue=defaultParamValues.get("gamma", None)))
        # parameterList.append(DiscreteParameter(name="max_depth", bounds=(1, 25), mutationRate=3, defaultValue=defaultParamValues.get("max_depth", None)))
        parameterList.append(UniformParameter(name="min_child_weight", bounds=(0, 10), mutationRate=0.4, defaultValue=defaultParamValues.get("min_child_weight", None)))
        parameterList.append(UniformParameter(name="max_delta_step", bounds=(0, 5), mutationRate=0.4, defaultValue=defaultParamValues.get("max_delta_step", None)))
        parameterList.append(UniformParameter(name="subsample", bounds=(0, 1), mutationRate=0.4, defaultValue=defaultParamValues.get("subsample", None)))
        parameterList.append(UniformParameter(name="scale_pos_weight", bounds=(1, 10), mutationRate=0.4, defaultValue=defaultParamValues.get("scale_pos_weight", None)))        #"Parameters: { "scale_pos_weight" } might not be used" error
    elif variation == 'global':
        parameterList.append(UniformParameter(name="eta", bounds=(0, 1), mutationRate=None, defaultValue=defaultParamValues.get("eta", None)))
        parameterList.append(UniformParameter(name="gamma", bounds=(0, 10), mutationRate=None, defaultValue=defaultParamValues.get("gamma", None)))
        # parameterList.append(DiscreteParameter(name="max_depth", bounds=(1, 25), mutationRate=None, defaultValue=defaultParamValues.get("max_depth", None)))
        parameterList.append(UniformParameter(name="min_child_weight", bounds=(0, 10), mutationRate=None, defaultValue=defaultParamValues.get("min_child_weight", None)))
        parameterList.append(UniformParameter(name="max_delta_step", bounds=(0, 5), mutationRate=None, defaultValue=defaultParamValues.get("max_delta_step", None)))
        parameterList.append(UniformParameter(name="subsample", bounds=(0, 1), mutationRate=None, defaultValue=defaultParamValues.get("subsample", None)))
        parameterList.append(UniformParameter(name="scale_pos_weight", bounds=(1, 10), mutationRate=None, defaultValue=defaultParamValues.get("scale_pos_weight", None)))        #"Parameters: { "scale_pos_weight" } might not be used" error
    else:
        raise Exception("Unknown option variation={variation}.")

    return parameterList


if __name__== '__main__':
    # Quick test
    print(initializeXGBoostParameters())





