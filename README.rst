Infectious disease agent-based modeling simulator
=========================================
Agent-based models (ABMs) are computer simulations used to study complex interactions between agents, time, and environment. ABMs are stochastic models built from the bottom up as individual agents (such as people in epidemiology) are assigned with certain attributes and behaviours. These agents are then programmed to behave and interact with other agents and environment by simple rules. In addition to being an interactive tool to understand causal mechanisms, ABMs can help make real-world predictions when real-world data is used to fine-tune the parameters of the model.

The `Infectious disease agent-based modeling simulator`_ or (ID-ABM) simulator is an Apache2 licensed ABM simulator specific for infectious diseases. The ID-ABM's core is built on top of the Python 3 ABM library `Mesa <https://github.com/projectmesa/mesa>`_. The ID-ABM simulator contains a number of key features important to dynamically model infectious disease transmission and related epidemiologic measures. The key features are controlled by various adjustable model parameters. The large number of parameters enables one to model a wide spectrum of infectious diseases in a simulated population, as well as to model different scenarios such as implementation of various public health interventions, for a given infectious disease. This simulator is modular, hence adding new features or modifying existing features should be readily accessible.

Agents' states
------------
.. image:: https://github.com/kaionwong/infectious-disease-agent-based-modeling/blob/master/project_result/patient_states_diagram.png
   :width: 75%
   :height: 50%
   :scale: 100%
   :alt: Patient states diagram

The ID-ABM is based on a susceptible-infected-dead-recovered (SIDR) model, such that the possible primary health state of a given individual or agent includes:

* Susceptible

* Infectious

* Recovered

* Dead (due to studied infectious disease)

More specifically, agents tend to have additional or more detailed health or disease states. This is captured in the ID-ABM. The specific symptom states for an infectious agent include:

* Asymptomatic

* Mild symptom

* Severe symptom

* Critical symptom

The specific immunity states for a recovered agent include:

* Susceptible

* Immune

The specific complication states for a recovered agent include:

* No complication

* Mild complication

* Severe complication

The symptom state of an infectious agent affects his/her needs to be hospitalized or ventilated and the probability to die from a given infectious disease. The complication state of a recovered agent affects his/her needs to require ``DrugX`` as medical treatment.

Key features
------------
Key features of the ID-ABM's include:

* **Network graph**. Social network between agents are modelled in a network graph. Based on a time-dependent probability function, an infectious agent is capable to transmit the disease to any immediate neighboring agents who are susceptible. To visualize the network graph in real-time, set ``server = make_server(graphics_option='full')`` in ``visualize.py``.

* **Gamma probability distribution**. Gamma distribution is used to model the following time-dependent probabilities: probability infectious agent transmits the disease to a neighboring susceptible agent (``prob_spread_virus``), probability an infectious agent is recovered (``prob_recover``), probability an infectious agent dies from the disease (``prob_virus_kill_host``), probabilities an infectious agent changes his/her symptom states (``prob_infectious_no_symptom_maintained``, ``prob_infectious_no_to_mild_symptom``, ``prob_infectious_no_to_severe_symptom``, ``prob_infectious_no_to_critical_symptom``, ``prob_infectious_mild_symptom_maintained``, ``prob_infectious_mild_to_no_symptom``, ``prob_infectious_mild_to_severe_symptom``, ``prob_infectious_mild_to_critical_symptom``, ``prob_infectious_severe_symptom_maintained``, ``prob_infectious_severe_to_no_symptom``, ``prob_infectious_severe_to_mild_symptom``, ``prob_infectious_severe_to_critical_symptom``, ``prob_infectious_critical_symptom_maintained``, ``prob_infectious_critical_to_no_symptom``, ``prob_infectious_critical_to_mild_symptom``, ``prob_infectious_critical_to_severe_symptom``). Within the ``GammaProbabilityGenerator`` class, the ``shape``, ``scale``, and ``loc`` control the overall shape of the probability function in the x-axis (or timing) and y-axis (or magnitude of probability). Additionally, ``magnitude_multiplier`` is introduced in the class to allow for greater control and flexibility over the magnitude of probability (y-axis). **Note**: while the advantage of using gamma distribution to model probability provides greater control and flexibility, its direct drawback is to drastically increase the possible number of combinations of different parameter values, which could lead to extremely large search space and long computing time.

* **Simple probability**. Simple probability (between 0.0-1.0) is used to model the following probabilities: probabilities a recovered agent changes his/her complication states (``prob_recovered_no_to_mild_complication``, ``prob_recovered_no_to_severe_complication``, ``prob_recovered_mild_to_no_complication``, ``prob_recovered_mild_to_severe_complication``, ``prob_recovered_severe_to_no_complication``, ``prob_recovered_severe_to_mild_complication``) and probability a recovered agent gains immunity (``prob_gain_immunity``).

* **Clinical resources**. It includes conditions and checks to determine if an agent requires certain clinical resources via the ``ClinicalResource`` class and ``agent``'s functions ``try_use_hospital_bed()``, ``try_use_icu_bed()``, ``try_use_ventilator()``, and ``try_use_drugX()``. The maximum capacity and associated cost for each of these resources can be specified. For example, for ICU hospitalization, its maximum capacity is specified by ``icu_bed_capacity_as_percent_of_population`` and its cost per time unit specified by ``icu_bed_cost_per_day``.

* **Social distancing**. The ``SocialDistancing`` class allows for the implementation of social distancing as a public health intervention. The time period and intensity of social distancing are specified by ``time_period`` and ``edge_threshold``, respectively. More than one sets of social distancing intensity over different time periods can be specified in one class instantiation, such as

.. code-block:: bash

    # Intensity at 0.75 from time 26 to 89; at 0.25 from time 90 to 998
    self.social_distancing = SocialDistancing(1, self, edge_threshold=[0.75, 0.25],
                                              time_period=[(26, 90), (90, 999)], current_time=None,
                                              on_switch=True)
    
* **Vaccination**. The ``Vaccine`` class allows for the implementation of vaccine as a public health intervention. The probability to be vaccinated, time period, and success rate of the vaccine are specified by ``prob_vaccinated``, ``time_period``, and ``vaccine_success_rate``, respectively. More than one sets of vaccination probabilities and success rates over different time periods can be specified in one class instantiation, such as

.. code-block:: bash

    # Vaccination probability at 0.80 and vaccine success rate at 0.75 from time 10 to 29; vaccination      probability at 0.25 and vaccine success rate at 0.80 from time 30 to 49
    self.vaccine = Vaccine(1, self, agent=None, prob_vaccinated=[0.80, 0.25],
                           vaccine_success_rate=[0.75, 0.80], time_period=[(10, 30), (30, 50)],
                           current_time=None, on_switch=True)

* **Testing**. The ``Testing`` class allows for the implementation of disease testing as a disease monitoring strategy. This is important since in the real world, the reported cases are the cases that have been tested and verified. Thus, these figures are only indirect indicators of the underlying true cases (including those infected cases not tested or reported). A minimal time unit required to pass before a subsequent test can be administered is specified in ``_min_days_between_two_tests``. The probability to be tested based on an agent's symptom state, time period, test sensitivity, test specificity are specified by ``prob_tested_for_no_symptom``, ``prob_tested_for_mild_symptom``, ``prob_tested_for_severe_symptom``, ``prob_tested_for_critical_symptom``, ``time_period``, ``test_sensitivity``, ``test_specificity``, specifically. More than one sets of symptom-specific test probabilities, sensitivity, and specificity over different time periods can be specified in one class instantiation, such as

.. code-block:: bash

    # Different sets of value between time 0 to 24, time 25 to 59, and time 60 to 998 
    self.testing = Testing(1, self, agent=None,
                           prob_tested_for_no_symptom=[0.005, 0.01, 0.01],
                           prob_tested_for_mild_symptom=[0.005, 0.01, 0.01],
                           prob_tested_for_severe_symptom=[0.01, 0.03, 0.05],
                           prob_tested_for_critical_symptom=[0.01, 0.03, 0.05],
                           test_sensitivity=[0.89, 0.95, 0.95], test_specificity=[0.95, 0.99, 0.99],
                           time_period=[(0, 25), (25, 60), (60, 999)], current_time=None, on_switch=True)


* **Modifiable probabilities**. An agent's ``age`` and existing comorbid conditions such as ``comorbid_hypertension``, ``comorbid_diabetes``, ``comorbid_ihd``, ``comorbid_asthma``, ``comorbid_cancer``, as well as whether or not they are receiving the necessary care (i.e., ``UseHospitalBedState``, ``UseICUBedState``, and ``UseVentilatorState``) can influence his/her probabilities to change symptom states, recover, or die from the disease. The associated rules of how these risk factors may modify these probabilities are controlled and stated within the function ``update_probability_by_special_condition()``. 

* **Epidemiologic measures**. The tracked epidemiology measures include:

    - Daily count/figure: ``Mean R0``, ``Test done``, ``Susceptible``, ``Infectious``, ``Recovered``, ``Dead``, ``Test-confirmed infectious``, ``Test-confirmed dead``, ``Infectious-no symptom``, ``Infectious-mild symptom``, ``Infectious-severe symptom``, ``Infectious-critical symptom``, ``Infectious using non-ICU hospital bed``, ``Infectious using ICU hospital bed``, ``Infectious using ventilator``, ``Recovered-no complication``, ``Recovered-mild complication``, ``Recovered-severe complication``, and ``Recovered using DrugX``.

    - Cumulative count: ``Cumulative test done``, ``Cumulative infectious``, ``Cumulative dead``, ``Cumulative test-confirmed infectious``, and ``Cumulative test-confirmed dead``.

    - Rate: ``Rate per 1M cumulative test done``, ``Rate per 1M cumulative infectious``, ``Rate per 1M cumulative dead``, ``Rate per 1M cumulative test-confirmed infectious``, and ``Rate per 1M cumulative test-confirmed dead``.

Additional features
------------

* In ``agent.py``, positive integer input for ``_stop_timer`` indicates when the simulation will stop, if ``None``, the simulation will run continuously.

* In ``network.py``, if a random seed is specified in ``set_network_seed``, the structure and connections of the network graph will remain the same even when the network is ``reset`` in ``run_single.py``, if ``None``, new structure and connections for a network graph will be randomly generated when it is ``reset``. If a random seed is specified in ``set_initial_infectious_node_seed``, the same agents will be assigned as initial infectious agents even when the network is reset, if ``None``, new agents will be randomly assigned as initial infectious agents when the network is ``reset``.

* When ``run_single.py`` is run, it activates the local server created in the ``visualize.py`` file. This creates and launches an interactive and "real-time" model visualization, using a server with JavaScript interface. The amount of graphics to be displayed can be specified by the ``graphics_option`` parameter from the ``make_server()`` function.

* Batch simulation runs can be done by configuring and executing the ``run_batch.py``. Each key (corresponding to the variable name of model parameter) within the ``br_params`` dictionary takes a list value. The list can take a single numeric value or multiple numeric values. When multiple numeric values are specified for a key, for examples ``'num_nodes': [1000, 5000, 10000]`` or ``'prob_spread_virus_gamma_shape': [1, 2, 3]``, all the combinations of specified parameter values will be conducted and recorded in a batch run. The ``num_iterations`` configures how many iterations each of the simulation run will be repeated. The ``start_date`` determines when the real-world (Alberta) data begins, as well as the date to be assigned as time (t) = 1 for the simulation. The ``num_max_steps_in_reality`` signals how many t unit (i.e., days) will be read as the end of the real-world data, while the ``num_max_steps_in_simulation`` signals how many t unit will be executed as the end of the simulation run. When ``num_max_steps_in_simulation`` is greater than ``num_max_steps_in_reality``, the difference in t unit is the total duration of time the simulation can help make future predictions in a real-world setting.

Demonstration of batch runs using both simulated and real-world data
------------
The use of the ID-ABM is demonstrated for Covid-19 in Alberta, Canada in 2020.

* **Overall steps**. 1) Construction of the ID-ABM, 2) Parameter search and validation, and 3) Real-world predictions. After the ID-ABM codebase was developed, parameter values specific to the current Covid-19 epidemic in Alberta are searched via an iterative manual and batch search (from ``run_single.py`` and ``run_batch.py``, respectively). The identified set of parameter values will be incorporated in the ID-ABM to simulate epidemiologic measures and time-series. The time-series of Rate per 1M cumulative test done, Rate per 1M cumulative test-confirmed infectious, and Rate per 1M cumulative test-confirmed dead between the real-world Alberta data and simulated data will be compared statistically via the Granger Causality test and Pearson correlation. Finally, the tested parameter value sets will be used to make prediction based on dynamics of various public health intervention to be implemented.

* **Alberta (and Canadian) data**. The ``probability.py`` includes the published age distribution and age- and sex-specific prevalence of asthma and cancer in Alberta, and the age- and sex-specific prevalence of hypertension, diabetes, and ischemic heart disease in Canada. These can be readily swapped with data published for other locations when ID-ABM is applied elsewhere. These real-world statistics are used to generate the demographic and comorbidity characteristics of the simulated agents during their instantiation in ``HostAgent`` class' ``__init__()``. When ``get_covid19_data.py`` is executed, it downloads the most up-to-date historical Covid-19 epidemiologic data across Canada from `COVID-19 Canada Open Data Working Group <https://github.com/ishaberry/Covid19Canada>`_. The gathered Covid-19 data is further filtered and processed in ``run_batch.py``. 

* **Parameter search and validation**. The Rate per 1M cumulative test done, Rate per 1M cumulative test-confirmed infectious cases, and Rate per 1M cumulative test-confirmed dead cases from this real-world aggregated Alberta Covid-19 data will be used to guide, validate, and finalize the parameter value sets that have statistically-significant (p<0.05 in Granger Causality and Pearson correlation tests) predictive quality on the corresponding real-world time-series. Since a brute force search of all possible parameter values for all the parameters will incur voluminous (and unmanageable) amount of combinations, a hybrid search using manual search/examination with a narrower parameter range of batch runs is conducted. (**Note**: For simulating complex models in a formal research or application setting, due to large computational effort to search parameter space, it is `recommended by Venkatramanan et al. (2018) <https://reader.elsevier.com/reader/sd/pii/S1755436517300221?token=EFD0DDB552C66746C44CEAE3E9D3349037A54BCA2C3FBA5C2D73C823B606391A6DB13BD91C76B4C878A1284ECC7E9881>`_ to incorporate a more formal optimization or Bayesian framework for parameter value search.) In order to statistically validate the simulated results (Rate per 1M cumulative test done, Rate per 1M cumulative test-confirmed infectious cases, and Rate per 1M cumulative test-confirmed dead cases) against the real-world Alberta data, Granger causality test and Pearson correlation are conducted (in ``run_batch.py``) to demonstrate that the simulated time series provided additional statistically-significant predictive quality as well as significant correlation as compared to the real-world time series.

* **Real-world predictions**. The graphs below include Predicted cumulative count of infected cases Alberta and Predicted cumulative count of deaths in Alberta due to Covid-19. These graphs were created by ``prediction_graphs.py``. The brown line prior to 2020-05-26 were real published data from Alberta. The multicolored lines from 2020-05-26 onward included projected simulations overlaid on the last date of the real-world data (or 2020-05-25). Specifically, the percent changes (in ``Cumulative test-confirmed infectious`` and ``Cumulative test-confirmed dead``) for the next day were calculated based on the simulated runs, via ``predict_by_percent_change_of_another_col()`` in ``run_batch.py``. The predicted figure for 2020-05-26 is calculated by multiplying the 2020-05-25 (last real-world) figure by the daily percent change between 2020-05-25 and 2020-05-26 derived from the simulation. The predicted figure for 2020-05-27 is calculated by multiplying the 2020-05-26 (first predicted) figure by the daily percent change between 2020-05-26 and 2020-05-27 derived from the simulation, and so on and so forth.

.. image:: https://github.com/kaionwong/infectious-disease-agent-based-modeling/blob/master/project_result/Graph_Predicted%20cumulative%20count%20of%20deaths%20in%20Alberta%20(Demo%20only).png
   :width: 100%
   :scale: 90%
   :alt: Figure – Predicted cumulative count of infected Covid-19 cases in Alberta in 2020 (Demo only)
*Above: 192 predictions generated by 192 different parameter combinations (more detail in ``\project_result\parameter_settings_for_batch_run.py``. Each parameter combination run was done with ``Total N`` at 10,000. Sample data files include ``\project_result\ disease_model_merged_data_vFinal_p0.csv*

.. image:: https://github.com/kaionwong/infectious-disease-agent-based-modeling/blob/master/project_result/Graph_Predicted%20cumulative%20count%20of%20infected%20cases%20in%20Alberta%20(Demo%20only).png
   :width: 100%
   :scale: 90%
   :alt: Figure – Predicted cumulative count of dead Covid-19 cases in Alberta in 2020 (Demo only)
*Above: 192 predictions generated by 192 different parameter combinations (more detail in ``\project_result\parameter_settings_for_batch_run.py``. Each parameter combination run was done with ``Total N`` at 10,000. Sample data files include ``\project_result\ disease_model_merged_data_vFinal_p0.csv*

Demonstration of single runs generating purely simulated data
------------
Three simulated runs were shown below to demonstrate the effects of social distancing and vaccination in a simulated population (N=1,000). For more details on specific parameter settings, see ``\project_result\parameter_settings_for_single_run.txt``.

* **Simulated Run #1 at time 0**.
.. image:: https://github.com/kaionwong/infectious-disease-agent-based-modeling/blob/master/project_result/screenshot_run1_t0.png
   :width: 85%
   :scale: 50%

* **Simulated Run #1 at time 60**.
.. image:: https://github.com/kaionwong/infectious-disease-agent-based-modeling/blob/master/project_result/screenshot_run1_t60.png
   :width: 85%
   :scale: 50%

* **Simulated Run #2 at time 60 (with social distancing starting at time 10)**.
.. image:: https://github.com/kaionwong/infectious-disease-agent-based-modeling/blob/master/project_result/screenshot_run2_t60_withSocialDistancing_vShort.png
   :width: 85%
   :scale: 50%

* **Simulated Run #3 at time 60 (with social distancing starting at time 10, and vaccination starting at time 20)**.
.. image:: https://github.com/kaionwong/infectious-disease-agent-based-modeling/blob/master/project_result/screenshot_run3_t60_withSocialDistancingAndVaccine_vShort.png
   :width: 85%
   :scale: 50%

References
------------
* Venkatramanan S, Lewis B, Chen J, et al. Using data-driven agent-based models for forecasting emerginginfectious diseases. Epidemics 2018;22:43-9.
