import copy
import time
from abc import ABC, abstractmethod
from typing import Dict, Any

pp = print
import numpy as np
import random
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import matplotlib.pyplot as plt

np.set_printoptions(linewidth=10000, precision=4)
import pandas as pd

pd.set_option("display.max_rows", None, "display.max_columns", None)
pd.set_option("display.width", 1000)
print("gridup3 loaded")
import matplotlib
import itertools


def legend_if_exists(ax):
    some_label = False
    for child in ax.get_children():
        # noinspection PyUnresolvedReferences
        if isinstance(child, matplotlib.lines.Line2D):
            lab = child.get_label()
            if lab[0] != "_":
                some_label = True
                break
    if some_label:
        ax.legend()


"""
Seulement 3 méthodes obligatoires pour un agent
"""


class UpAgent(ABC):
    @abstractmethod
    def fit_train_case(self, case) -> bool:
        # called serveral times, until period end, on the train-cases
        # if False is returned, the training period is interupt
        raise Exception("no train_step  method")

    @abstractmethod
    def score_test_cases(self) -> float:
        # called one time, at the end of the training
        # the score is used to sort result. Bigger score, better result
        raise Exception("no test_score_global  method")

    @abstractmethod
    def metrics_test_case(self, case) -> dict:
        # called one time, at the end of the training, on the test-cases
        raise Exception("no test_metrics method")

    # facultatif: une métrique globale
    def metrics_test_cases(self) -> dict or None:
        return None
        # raise Exception("no metrics_test_cases method")

    # faculatif, uniquement si l'on appelle ParamSearchResult.plot_train_cases
    def plot_train_case(self, ax, case, custom_args=None):
        raise Exception(
            "no plot_train_case method, while you have called ParamSearchResult.plot_train_cases"
        )

    # faculatif, uniquement si l'on appelle ParamSearchResult.plot_test_cases
    def plot_test_case(self, ax, case, custom_args=None):
        raise Exception(
            "no plot_test_case method, while you have called ParamSearchResult.plot_test_cases"
        )


def transform_period_for_each(period_for_each):
    if period_for_each.endswith("s"):
        period_for_each = period_for_each[:-1]
    if period_for_each.endswith("e"):
        period_for_each = period_for_each[:-1]

    if period_for_each.endswith("second"):
        period_unity = "second"
        period_duration = float(period_for_each[:-6].rstrip())

    elif period_for_each.endswith("minut"):
        period_unity = "second"
        period_duration = float(period_for_each[:-5].rstrip()) * 60

    elif period_for_each.endswith("step"):
        period_unity = "step"
        period_duration = float(period_for_each[:-5].rstrip())

    else:
        raise Exception("perdio must finish by second(e.s) or minut(e.s) or step(s)")

    return period_duration, period_unity


def plot_agent_predictions(agents, cases, agents_names=None, custon_param=None):
    if agents_names is not None:
        assert len(agents_names) == len(agents)
    else:
        agents_names = [""] * len(agents)  # pas de nom=> noms vides

    ni, nj = len(cases), len(agents)

    fig, axs = plt.subplots(ni, nj, figsize=(4 * nj, 2 * ni))
    if ni == 1 and nj == 1:
        axs = np.array([[axs]])
    elif ni == 1:
        axs = axs[None, :]
    elif nj == 1:
        axs = axs[:, None]

    for i, case in enumerate(cases):
        label_i = str(case)

        for j, (agent, agentName) in enumerate(zip(agents, agents_names)):
            ax: Any = axs[i, j]
            agent.plot_train_case(ax, case, custon_param)
            if j == 0:
                ax.set_ylabel(label_i)
            if i == ni - 1:
                ax.set_xlabel(agentName)
            if i == j == 0:
                legend_if_exists(ax)

    # fig.text(0.5, 0.01, f"agent.{self.paramName}", ha='center',fontsize=18)
    fig.text(0.01, 0.5, f"cases", va="center", rotation="vertical", fontsize=18)
    fig.tight_layout(w_pad=1, h_pad=1)

    plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.1)


"""Zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz"""
"""Zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz"""
"""Zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz"""
"""Zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz"""
"""Zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz"""


def check_float(a):
    assert isinstance(a, float) or isinstance(
        a, int
    ), f"This:'{a}' must be a float or an int"


class GeneralResult:
    def __init__(self, gU: "GridupTrainer"):
        self.content = {}
        self.gU = gU
        # value given after
        self.df = None

    def never_seen(self, param_value):
        hash_ = str(param_value)
        return self.content.get(hash_) is None

    def add(self, scoreName_case_val, globalMetricName_value, param_value, agent):
        self.content[str(param_value)] = (
            scoreName_case_val,
            globalMetricName_value,
            copy.deepcopy(param_value),
            agent,
        )

    def find_best(self):
        sorted_results = []  # sorted(self.content.values(), key=lambda a: a[0])
        "on ajoute un score issu de la moyenne de tous les scores sur les cases de train (option par défaut)"
        for (
            metricName_case_val,
            globalMetricName_value,
            param_value,
            agent,
        ) in self.content.values():
            score = -globalMetricName_value["accuracy"]  # agent.score_test_cases()
            score = float(score)
            if np.isnan(score):
                score = -float("inf")
            sorted_results.append(
                (score, metricName_case_val, globalMetricName_value, param_value, agent)
            )

        # best score are the highest, so they appear on the first line of the dataFrame
        sorted_results = sorted(sorted_results, key=lambda a: -a[0])

        """création d'une dataFrame de présentation des résultat"""
        df = pd.DataFrame()

        for i, (
            score,
            metricName_case_val,
            globalMetricName_value,
            param_value,
            agent,
        ) in enumerate(sorted_results):
            rank = i + 1
            check_float(score)
            df.loc[rank, "score"] = score
            for k, v in param_value.items():
                df.loc[rank, str(k)] = str(v)

            for metricName, case_val in metricName_case_val.items():
                for case, val in case_val.items():
                    case_ = (
                        str(case) + "(tr)"
                        if case in self.gU.cases_for_train
                        else str(case)
                    )
                    k = metricName + ":" + case_
                    df.loc[rank, k] = val

            for globalMetricName, val in globalMetricName_value.items():
                df.loc[rank, globalMetricName] = val

        # print(df)
        best = sorted_results[0]
        score, _, _, param_value, agent = best
        self.best_score = score
        self.best_agent = agent
        self.best_params = param_value
        self.df = df

        # return {param_value,agent,df}


class ParamSearchResult:
    def __init__(
        self,
        paramName,
        gU: "GridupTrainer",
        metricName_paramVal_case_value: dict,
        globalMetricName_paramVal_value: dict,
        agents: dict,
        agents_justScore,
    ):
        self.paramName = paramName
        self.gU = gU
        self.metricName_paramVal_case_value = metricName_paramVal_case_value
        self.globalMetricName_paramVal_value = globalMetricName_paramVal_value
        self.agents = agents
        self.agents_justScore = agents_justScore

        # value given after
        self.best_agent = None
        self.best_score = None
        self.df = None

    def compare_scores(self, x_are_cases=False, yscale_log=False):
        nb = len(self.gU.metric_names)
        fig, axs = plt.subplots(nb, 1, figsize=(10, 5 * nb))
        if nb == 1:
            axs = [axs]

        if x_are_cases:
            plot_func = plot_x_are_cases
        else:
            plot_func = plot_x_are_params

        for i, scoreName in enumerate(self.gU.metric_names):
            plot_func(
                axs[i],
                scoreName,
                self.metricName_paramVal_case_value[scoreName],
                self.paramName,
                self.gU.cases_for_train,
                yscale_log,
            )

        fig.tight_layout()

        self.fig = fig
        self.axs = axs

    def plot_train_cases(self, custon_param=None):
        self._plot_cases(True, custon_param)

    def plot_test_cases(self, custon_param=None):
        self._plot_cases(False, custon_param)

    def _plot_cases(self, is_train, custon_param=None):
        if self.paramName is None:
            raise Exception(
                "please call the method 'one_param_search' before 'plot_predictions' "
            )

        param_keys = self.paramName.split("+")
        if len(param_keys) == 1:
            paramValues = self.gU.agent_varying_params[self.paramName]
        else:
            paramValues = []
            for val0 in self.gU.agent_varying_params[param_keys[0]]:
                for val1 in self.gU.agent_varying_params[param_keys[1]]:
                    paramValues.append((val0, val1))

        cases = self.gU.cases_for_train if is_train else self.gU.cases_for_test

        ni, nj = len(cases), len(paramValues) + len(self.gU.agents_justPlotPrediction)

        fig, axs = plt.subplots(ni, nj, figsize=(4 * nj, 2 * ni))
        if ni == 1 and nj == 1:
            axs = np.array([[axs]])
        elif ni == 1:
            axs = axs[None, :]
        elif nj == 1:
            axs = axs[:, None]

        for i, case in enumerate(cases):
            label_i = str(case)
            if (
                case in self.gU.cases_for_train
            ):  # self.data_creators_train_dict.get(case) is not None:
                label_i += " (tr)"
            for j, paramValue in enumerate(paramValues):
                ax: Any = axs[i, j]

                if self.agents.get(paramValue) is None:
                    print(
                        f"no agent for {self.paramName}:{paramValue}, perhaps it is excluded by the functions give by the arg 'excluded' of ther trainer"
                    )
                else:
                    if is_train:
                        agent = self.agents[paramValue]
                        agent.plot_train_case(ax, case, custon_param)
                    else:
                        self.agents[paramValue].plot_test_case(ax, case, custon_param)

                if j == 0:
                    ax.set_ylabel(label_i)
                if i == ni - 1:
                    ax.set_xlabel(self.paramName + ":" + str(paramValue))
                if i == j == 0:
                    legend_if_exists(ax)

            for k, (agentName, agentJustPlot) in enumerate(
                self.gU.agents_justPlotPrediction.items()
            ):
                # noinspection PyTypeChecker
                ax = axs[i, k + len(paramValues)]
                if is_train:
                    agentJustPlot.plot_train_case(ax, case, custon_param)
                else:
                    agentJustPlot.plot_test_case(ax, case, custon_param)

                if i == ni - 1:
                    ax.set_xlabel(agentName)

            # fig.text(0.5, 0.01, f"agent.{self.paramName}", ha='center',fontsize=18)
            fig.text(0.01, 0.5, f"cases", va="center", rotation="vertical", fontsize=14)
            fig.tight_layout(w_pad=1, h_pad=1)

            plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.25)


class GridupTrainer:
    def __init__(
        self,
        create_agent_fn,  # un constructeur d'agent, ou bien une fonction qui renvoie un agent, c'est idem
        agent_params: dict,
        agent_const: dict,
        train_duration_for_one_agent: str,
        cases_for_train: list,
        cases_for_test: list,
        *,
        verbose=False,
        agents_justScore: Dict[str, UpAgent] = {},
        agents_justPlotPrediction: Dict[str, UpAgent] = {},
        try_testing_before_training=False,
        excluded=None,
    ):
        print("Start GridUp trainer")

        self.agent_fixed_params = {k: agent_params[k][0] for k in agent_params.keys()}
        self.excluded = excluded

        """si la valeur par défaut du paramètre est aussi présente dans les choix, 
        il ne faut pas la répeter 2 fois dans le choix. D'où le list(set(...)) ci-dessous """
        self.agent_varying_params = {
            k: sorted(list(set([agent_params[k][0]] + agent_params[k][1])))
            for k in agent_params.keys()
        }

        self.params_for_search = []
        for k, v in agent_params.items():
            if len(v) == 3:
                assert (
                    v[2] is True
                ), f"Pour le param={k}, le troisième élément du triplet doit être 'True'"
                self.params_for_search.append(k)

        self.agent_const = agent_const
        self.create_agent_fn = create_agent_fn

        self.train_duration_for_one_agent = train_duration_for_one_agent
        self.cases_for_train = cases_for_train
        self.cases_for_test = cases_for_test
        # *

        self.verbose = verbose
        self.agents_justScore = agents_justScore
        self.agents_justPlotPrediction = agents_justPlotPrediction

        self.period_duration, self.period_duration_unity = transform_period_for_each(
            train_duration_for_one_agent
        )

        if try_testing_before_training:
            self._test_an_agent()

    def is_excluded(self, param):
        if self.excluded is None:
            return False
        for exclu_fn in self.excluded:
            if exclu_fn(param):
                if self.verbose:
                    print(f"\t\t\t Excluded: {param}")
                return True
        return False

    def random_search(self, minutes) -> GeneralResult:
        print("\nRANDOM SEARCH")

        def random_dico():
            model_params_random = {}
            for param_name in self.agent_varying_params.keys():
                param_value = random.choice(
                    self.agent_varying_params[param_name]
                )  # la procédure numpy : np.random.choice(self.agent_varying_params[param_name]) ne fonctionne pas quand les param_values sont des tuples
                model_params_random[param_name] = param_value
            return model_params_random

        return self._random_search(minutes, random_dico)

    def random_search_around(self, minutes, proba: float) -> GeneralResult:
        print("\nRANDOM SEARCH AROUND")
        variation_proba = {k: proba for k in self.agent_fixed_params.keys()}

        def random_dico():
            model_params_random = {}
            for param_name in self.agent_varying_params.keys():
                if np.random.rand() < variation_proba[param_name]:
                    param_value = random.choice(
                        self.agent_varying_params[param_name]
                    )  # la procédure numpy : np.random.choice(self.agent_varying_params[param_name]) ne fonctionne pas quand les param_values sont des tuples
                else:
                    param_value = self.agent_fixed_params[param_name]
                model_params_random[param_name] = param_value
            return model_params_random

        return self._random_search(minutes, random_dico)

    def _random_search(self, minutes, random_dico) -> GeneralResult:
        generalResult = GeneralResult(self)

        ti0 = time.time()
        OK = True
        try:
            while OK:
                OK = time.time() - ti0 < minutes * 60
                param_value = random_dico()
                if self.verbose:
                    print("\t looking at:", param_value)

                if generalResult.never_seen(param_value):
                    if not self.is_excluded(param_value):
                        agent: UpAgent = self.create_agent_fn(
                            **{**param_value, **self.agent_const}
                        )
                        (
                            mean_train_step_duration,
                            scoreName_case_val,
                            globalMetricName_value,
                        ) = self.train(agent)

                        generalResult.add(
                            scoreName_case_val,
                            globalMetricName_value,
                            param_value,
                            agent,
                        )
                else:
                    if self.verbose:
                        print("\t\t already seen")

        except KeyboardInterrupt:
            print(f"random search interupt after {time.time()-ti0} seconds")

        generalResult.find_best()

        return generalResult

    def _test_an_agent(self):
        self.duration_infos = {}

        if self.agent_fixed_params is None:
            raise Exception("no agent_fixed_params are given")

        ti0 = time.time()
        agent = self.create_agent_fn(**{**self.agent_fixed_params, **self.agent_const})
        self.duration_infos["agent creation"] = time.time() - ti0
        ti0 = time.time()
        score = agent.score_test_cases()
        check_float(score)
        self.duration_infos["score"] = time.time() - ti0

        metrics_duration = {}
        for case in self.cases_for_test:
            ti0 = time.time()
            metrics = agent.metrics_test_case(case)
            metrics_duration[case] = time.time() - ti0

            if metrics is None or not isinstance(metrics, dict):
                raise Exception("la méthode agent.metric doit renvoyer un dictionnaire")

        self.duration_infos["metrics by case"] = metrics_duration

        print("grid_up durations info:", self.duration_infos)

        return agent

    def make_an_agent(self, test_its_method=False) -> UpAgent:
        agent = self.create_agent_fn(**{**self.agent_fixed_params, **self.agent_const})
        if test_its_method:
            score = agent.score_test_cases()
            print("score:", score)
            for case in self.cases_for_test:
                metrics = agent.metrics_test_case(case)
                print(f"for case:{case}, metrics are:{metrics}")
        return agent

    def param_search(self):
        nb_triplet = len(self.params_for_search)
        if nb_triplet == 0:
            raise Exception(
                f"you must indicate which param to serach with one or two triplets, here we have: {nb_triplet} triplet "
            )
        elif nb_triplet == 1:
            return self.one_param_search(self.params_for_search[0])
        else:
            return self.two_or_more_param_search(self.params_for_search)

    #
    # def two_param_search(self,paramNames:list)->ParamSearchResult:
    #     print("\nTWO PARAMS SEARCH")
    #     assert len(paramNames)==2
    #
    #     paramNameComposed=paramNames[0]+"+"+paramNames[1]
    #     if self.verbose:
    #         print(f"\t varying '{paramNameComposed}' in {self.agent_varying_params[paramNames[0]]}x{self.agent_varying_params[paramNames[1]]} . Train with data-cases:{list(self.cases_for_train)}, Test with data-cases:{list(self.cases_for_test)}")
    #         print(f"\t Other paramers are fixed:",[(k,v) for k,v in self.agent_fixed_params.items() if k not in paramNames])
    #
    #     essaies = []
    #     for paramVal0 in self.agent_varying_params[paramNames[0]]:
    #         for paramVal1 in self.agent_varying_params[paramNames[1]]:
    #             params = {**self.agent_fixed_params, paramNames[0]: paramVal0, paramNames[1]: paramVal1}
    #             essaies.append((params, (paramVal0,paramVal1)))
    #
    #     return self.several_param_search(paramNameComposed, essaies)

    def two_or_more_param_search(self, paramNames: list) -> ParamSearchResult:
        print("\nSEVERAL PARAMS SEARCH")
        assert set(paramNames).issubset(
            set(self.agent_varying_params.keys())
        ), f"param to seach:{paramNames} must be a subset of all varying parameters:{list(self.agent_varying_params.keys())}"
        # assert len(paramNames)==2
        paramNameComposed = str(paramNames)

        if self.verbose:
            print(
                f"\t varying '{paramNameComposed}'. Train with data-cases:{list(self.cases_for_train)}, Test with data-cases:{list(self.cases_for_test)}"
            )
            print(
                f"\t Other paramers are fixed:",
                [
                    (k, v)
                    for k, v in self.agent_fixed_params.items()
                    if k not in paramNames
                ],
            )

        """
        ex: 
        paramNames=[name0,name1]
        self.agent_varying_params={name0:[1,2],name1:[A,B]}
        on veut: 
        [(name0,1),(nam1,A)], [(name0,1),(nam1,B)], [(name0,2),(nam1,A)],[(name0,2),(nam1,B)]
        C'est le produit cartésien de 
        (name0,1),(name0,2) par (name1,A),(name1,B)
        """
        set_to_multiply = []
        for paramName in paramNames:
            a_set = []
            set_to_multiply.append(a_set)
            for val in self.agent_varying_params[paramName]:
                a_set.append((paramName, val))

        essaies = []
        for element in itertools.product(*set_to_multiply):
            dico = {k: v for (k, v) in element}
            key = str(list(dico.values()))[1:-1]  # on peut prendre aussi key=str(dico)
            # on complète le dico avec les param qui ne bougent pas dans ce search
            for k, v in self.agent_fixed_params.items():
                if k not in paramNames:
                    dico[k] = v
            essaies.append((dico, key))

        #
        # for paramVal0 in self.agent_varying_params[paramNames[0]]:
        #     for paramVal1 in self.agent_varying_params[paramNames[1]]:
        #         params = {**self.agent_fixed_params, paramNames[0]: paramVal0, paramNames[1]: paramVal1}
        #         essaies.append((params, (paramVal0,paramVal1)))

        return self._several_param_search(paramNameComposed, essaies)

    def one_param_search(self, paramName: str) -> ParamSearchResult:
        print("\nONE PARAM SEARCH")
        paramValues_changing = self.agent_varying_params[paramName]
        if self.verbose:
            print(
                f"\t varying '{paramName}' in {paramValues_changing}. Train with data-cases:{list(self.cases_for_train)}, Test with data-cases:{list(self.cases_for_test)}"
            )
            print(
                f"\t Other paramers are fixed:",
                [(k, v) for k, v in self.agent_fixed_params.items() if k != paramName],
            )

        essaies = []
        for paramVal0 in paramValues_changing:
            params = {**self.agent_fixed_params, paramName: paramVal0}
            essaies.append((params, paramVal0))

        return self._several_param_search(paramName, essaies)

    def _several_param_search(self, paramName: str, essaies) -> ParamSearchResult:
        if self.agent_fixed_params is None:
            raise Exception("no agent_fixed_params are given")

        generalResult = GeneralResult(self)
        metricName_paramVal_case_value = None
        globalMetricName_paramVal_value = None

        agents = {}
        for param_value, key in essaies:
            if not self.is_excluded(param_value):
                if self.verbose:
                    print(f"\t\t train agent with: {paramName}={key}")

                # self.score_names est construit dans la méthode suivante

                agent: UpAgent = self.create_agent_fn(
                    **{**param_value, **self.agent_const}
                )
                (
                    mean_train_step_duration,
                    metricName_case_val,
                    globalMetricName_val,
                ) = self.train(agent)

                if (
                    metricName_paramVal_case_value is None
                ):  # la première fois uniquement
                    metricName_paramVal_case_value = {k: {} for k in self.metric_names}
                if globalMetricName_paramVal_value is None:
                    globalMetricName_paramVal_value = {
                        k: {} for k in self.metricGobal_names
                    }

                agents[key] = agent

                generalResult.add(
                    metricName_case_val, globalMetricName_val, param_value, agent
                )

                for metricName in self.metric_names:
                    metricName_paramVal_case_value[metricName][
                        key
                    ] = metricName_case_val[metricName]
                for globalMetricName in self.metricGobal_names:
                    globalMetricName_paramVal_value[globalMetricName][
                        key
                    ] = globalMetricName_val[globalMetricName]

        oneParamResult = ParamSearchResult(
            paramName,
            self,
            metricName_paramVal_case_value,
            globalMetricName_paramVal_value,
            agents,
            self.agents_justScore,
        )
        generalResult.find_best()

        oneParamResult.best_agent = generalResult.best_agent
        oneParamResult.best_score = generalResult.best_score
        oneParamResult.df = generalResult.df

        return oneParamResult

    def test_on_agent_justScore(self):
        print("TEST ON SPECIAL AGENTS (non trainable, just for test score)")
        df = pd.DataFrame()

        for agentName, agent in self.agents_justScore.items():
            metricName_case_value, _ = self.test(agent, None)
            for scoreName, case_val in metricName_case_value.items():
                for case, val in case_val.items():
                    case_ = (
                        str(case) + "(tr)"
                        if case in self.cases_for_train
                        else str(case)
                    )
                    k = scoreName + ":" + case_
                    df.loc[agentName, k] = val
        print(df)

    def train(self, agent: UpAgent):
        agent_train_step_durations = []
        step_count = 0
        best_score = 1e18
        try:
            ok = True
            ti0 = time.time()
            while ok:
                step_count += 1
                if self.period_duration_unity == "second":
                    ok = time.time() - ti0 < self.period_duration
                else:
                    ok = step_count < self.period_duration

                ti1 = time.time()
                """C'est important d'alterner les train-cases. 
                sinon l'apprentissage initial se concenter sur le premier train-case, le final sur le dernier"""
                stop = False
                for case in self.cases_for_train:
                    status = agent.fit_train_case(case)
                    scoreName_case_val, globalMetricName_value = self.test(agent, 0.0)
                    # print(globalMetricName_value)
                    if globalMetricName_value["accuracy"] < best_score:
                        best_score = globalMetricName_value["accuracy"]
                        best_scoreName_case_val, best_globalMetricName_value = (
                            scoreName_case_val,
                            globalMetricName_value,
                        )
                    stop = status is not None and not status
                    if stop:
                        break
                agent_train_step_durations.append(time.time() - ti1)
                if stop:
                    if self.verbose:
                        print(
                            f"\t\t\t interuption de l'entrainement de l'agent car le train_step a renvoyer la valeur 'False'"
                        )
                    break

            if self.period_duration_unity == "second":
                if self.verbose:
                    print(
                        f"\t\t\t{step_count} train_step par 'case' effectués en {time.time()-ti0} secondes"
                    )

        except KeyboardInterrupt:
            pass

        mean_train_step_duration = (
            np.mean(agent_train_step_durations)
            if len(agent_train_step_durations) > 0
            else 0.0
        )
        # if self.verbose:
        #     print("\t\t mean_train_step_duration:",mean_train_step_duration)
        if (
            self.period_duration_unity == "second"
            and mean_train_step_duration > 0.2 * self.period_duration
        ):
            print(
                f"/!\ la durée d'un agent.train_step (pour tous les train-cases) vaut {mean_train_step_duration}, et elle n'est non négligeable par rapport à la durée totale d'une période={self.period_duration}"
            )
        best_globalMetricName_value[
            "mean_train_step_duration"
        ] = mean_train_step_duration
        print(best_globalMetricName_value)

        return (
            mean_train_step_duration,
            best_scoreName_case_val,
            best_globalMetricName_value,
        )

    def test(self, agent: UpAgent, mean_train_step_duration):
        metricName_case_value = None
        assert self.cases_for_test is not None, "cases_for_test are None"
        cases = self.cases_for_test

        for i, case in enumerate(cases):
            metrics = agent.metrics_test_case(case)
            if metricName_case_value is None:
                self.metric_names = list(metrics.keys())
                metricName_case_value = {
                    metric_name: {} for metric_name in self.metric_names
                }
            for metric_name in metrics.keys():
                metricName_case_value[metric_name][case] = metrics[metric_name]

        globalMetricName_val = agent.metrics_test_case(0)
        if globalMetricName_val is None:
            globalMetricName_val = {}

        self.metricGobal_names = list(globalMetricName_val.keys())
        globalMetricName_val["mean_train_step_duration"] = mean_train_step_duration

        if self.verbose and mean_train_step_duration != 0.0:
            print(f"\t\t\tmetrics by case: {metricName_case_value}")
            print(f"\t\t\tmetrics  global: {globalMetricName_val}")

        return metricName_case_value, globalMetricName_val


def plot_x_are_cases(
    ax, scoreName, param_case_score, paramName, train_cases, yscale_log
):
    """
    res: param → (case → score)
    """

    one_case_score = list(param_case_score.values())[0]

    cases = list(one_case_score.keys())
    cases_are_str = isinstance(cases[0], str)

    cases_dico = None
    train_cases_are_str = isinstance(train_cases[0], str)

    if cases_are_str != train_cases_are_str:
        raise Exception("test and train cases are not of same type (str or scalar)")

    if cases_are_str:
        cases_num = range(len(cases))
        cases_str = cases
        cases_dico = {k: v for k, v in zip(cases_str, cases_num)}
    else:
        cases_num = cases
        cases_str = cases

    for param, dico in param_case_score.items():
        ax.plot(cases_num, list(dico.values()), ".-", label=str(param))

    for x in train_cases:
        if cases_are_str:
            num_val = cases_dico.get(x)
            if num_val is not None:
                ax.axvline(x=num_val, linestyle=":")
            else:
                print(
                    f"warning: the train-case '{x}' cannot be represented because it is a string that not belongs to the test-cases"
                )
        else:
            ax.axvline(x=x, linestyle=":")

    ax.set_title(f"data_case→{scoreName} by {paramName}")
    if yscale_log:
        ax.set_yscale("log")

    ax.set_xticks(
        cases_num,
    )
    rotation = "vertical" if len(str(cases_str[0])) >= 4 else None
    ax.set_xticklabels(cases_str, rotation=rotation)
    ax.set_xlabel("data cases")
    ax.set_ylabel(scoreName)
    ax.legend()


def inverse_dico(param_case_score):
    res = {}
    for param, case_score in param_case_score.items():
        for case, score in case_score.items():
            line = res.get(case, {})
            line[param] = score
            res[case] = line
    return res


def plot_x_are_params(
    ax, scoreName, param_case_score, paramName, train_cases, yscale_log
):
    case_param_score = inverse_dico(param_case_score)
    param_scores = list(case_param_score.values())[0]
    params = list(param_scores.keys())
    params_are_str = True  # isinstance(params[0],str)#TODO

    if params_are_str:
        params_num = range(len(params))
        params_str = params
    else:
        params_num = params
        params_str = params

    train_cases = set(train_cases)

    for case, param_score in case_param_score.items():
        label = str(case)
        if case in train_cases:
            label += "(tr)"
        ax.plot(params_num, list(param_score.values()), ".-", label=label)

    ax.set_title(f"{paramName}→{scoreName} by data_case")
    if yscale_log:
        ax.set_yscale("log")
    ax.set_xticks(params_num)
    rotation = "vertical" if len(str(params_str[0])) >= 4 else None
    ax.set_xticklabels(params_str, rotation=rotation)
    ax.set_xlabel(f"{paramName}")
    ax.set_ylabel(scoreName)
    ax.legend()


"""
#####################  TEST
"""


class AgentUltratoy(UpAgent):
    def __init__(self, add0, add1, add2, verbose):
        self.add0 = add0
        self.add1 = add1
        self.add2 = add2  # ne sert à rien

        self.wei0 = 0.0
        self.wei1 = 0.0

        self.nb_train_steps = 0
        if verbose:
            print("coucou")

    # obligatoire
    def fit_train_case(self, case):
        self.nb_train_steps += 1
        # ce n'est pas vraiment une optimization ici
        self.wei0 += self.add0
        self.wei1 += self.add1

    # obligatoire
    def score_test_cases(self):
        return self.wei0 + self.wei1

    # obligatoire
    def metrics_test_case(self, case):
        return {"wei0": self.wei0 * case, "wei1": self.wei1 * case}

    # obligatoire
    def metrics_test_cases(self) -> dict:
        return {"nb_train_steps": self.nb_train_steps}

    def plot_train_case(self, ax, case, custom_args=None):
        x = np.linspace(0, 1, 10)
        ax.plot(x, x * case)

    def plot_test_case(self, ax, case, custom_args=None):
        x = np.linspace(0, 1, 10)
        ax.plot(x, x * case)


def test_ultratoy():
    gU = GridupTrainer(
        AgentUltratoy,
        agent_params={
            "add0": (0, [-1, 1], True),
            "add1": (0, [-1, 1]),
            "add2": (0, [2, 10]),  # aucun effet
        },
        agent_const={"verbose": False},
        cases_for_train=[0.1, 0.3],
        cases_for_test=[0.1, 0.2, 0.3],
        train_duration_for_one_agent="1 steps",
        verbose=True,
        excluded=[lambda param: param["add0"] == param["add1"]],
    )

    generalResult = gU.random_search(1e-5)
    print(generalResult.df)

    radomAroundResult = gU.random_search_around(1e-5, proba=0.3)
    print(radomAroundResult.df)

    oneParamResult = gU.one_param_search("add0")
    print(oneParamResult.df)
    oneParamResult.compare_scores()
    oneParamResult.plot_test_cases()
    oneParamResult.plot_train_cases()

    result = gU.two_or_more_param_search(["add0", "add1", "add2"])
    print(result.df)
    result.compare_scores()

    result = gU.one_param_search("add0")
    result.compare_scores(x_are_cases=True)

    plt.show()


if __name__ == "__main__":
    test_ultratoy()
