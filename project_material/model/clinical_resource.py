from mesa import Agent

class ClinicalResource(Agent):
    def __init__(self, unique_id, model,
                 hospital_bed_capacity_as_percent_of_population, hospital_bed_cost_per_day, hospital_bed_current_load,
                    hospital_bed_use_day_tracker,
                 icu_bed_capacity_as_percent_of_population, icu_bed_cost_per_day, icu_bed_current_load,
                    icu_bed_use_day_tracker,
                 ventilator_capacity_as_percent_of_population, ventilator_cost_per_day, ventilator_current_load,
                    ventilator_use_day_tracker,
                 drugX_capacity_as_percent_of_population, drugX_cost_per_day, drugX_current_load,
                    drugX_use_day_tracker,
                 ):
        assert ValueError (icu_bed_capacity_as_percent_of_population <= hospital_bed_capacity_as_percent_of_population), \
                '`icu_bed_capacity_as_percent_of_population` can not be greater than `hospital_bed_capacity_as_percent_of_population`.'
        super().__init__(unique_id, model)
        self.total_hospital_bed = int(hospital_bed_capacity_as_percent_of_population * self.model.num_nodes)
        self.hospital_bed_cost_per_day = hospital_bed_cost_per_day
        self.hospital_bed_use_day_tracker = hospital_bed_use_day_tracker
        self.hospital_bed_current_load = hospital_bed_current_load
        self.hospital_bed_maxed_out = False
        self.total_hospital_bed_related_cost = None

        self.total_icu_bed = int(icu_bed_capacity_as_percent_of_population * self.model.num_nodes)
        self.icu_bed_cost_per_day = icu_bed_cost_per_day
        self.icu_bed_use_day_tracker = icu_bed_use_day_tracker
        self.icu_bed_current_load = icu_bed_current_load
        self.icu_bed_maxed_out = False
        self.total_icu_bed_related_cost = None

        self.total_ventilator = int(ventilator_capacity_as_percent_of_population * self.model.num_nodes)
        self.ventilator_cost_per_day = ventilator_cost_per_day
        self.ventilator_use_day_tracker = ventilator_use_day_tracker
        self.ventilator_current_load = ventilator_current_load
        self.ventilator_maxed_out = False
        self.total_ventilator_related_cost = None

        self.total_drugX = int(drugX_capacity_as_percent_of_population * self.model.num_nodes)
        self.drugX_cost_per_day = drugX_cost_per_day
        self.drugX_use_day_tracker = drugX_use_day_tracker
        self.drugX_current_load = drugX_current_load
        self.drugX_maxed_out = False
        self.total_drugX_related_cost = None

    def check_available_hospital_bed(self):
        if self.total_hospital_bed > self.hospital_bed_current_load:
            self.hospital_bed_maxed_out = False
            return True
        else:
            self.hospital_bed_maxed_out = True
            return False

    def check_available_icu_bed(self):
        if self.total_icu_bed > self.icu_bed_current_load:
            self.icu_bed_maxed_out = False
            return True
        else:
            self.icu_bed_maxed_out = True
            return False

    def check_available_ventilator(self):
        if self.total_ventilator > self.ventilator_current_load:
            self.ventilator_maxed_out = False
            return True
        else:
            self.ventilator_maxed_out = True
            return False

    def check_available_drugX(self):
        if self.total_drugX > self.drugX_use_day_tracker:
            self.drugX_maxed_out = False
            return True
        else:
            self.drugX_maxed_out = True
            return False