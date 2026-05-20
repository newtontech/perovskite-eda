/share/yhm/test/20260216_with_chemical_merge_fast/merged_llm_crossref_data_streaming_with_chemical_data_fast.xlsx has the following columns:



cas_number	pubchem_id	smiles	molecular_formula	molecular_weight	h_bond_donors	h_bond_acceptors	rotatable_bonds	tpsa	log_p


as chemical data,

jv_reverse_scan_pce_without_modulator	jv_reverse_scan_j_sc_without_modulator	jv_reverse_scan_v_oc_without_modulator	jv_reverse_scan_ff_without_modulator	jv_reverse_scan_pce	jv_reverse_scan_j_sc	jv_reverse_scan_v_oc	jv_reverse_scan_ff	jv_hysteresis_index_without_modulator	jv_hysteresis_index

as JV data, perticularly, for jv_reverse_scan_pce - jv_reverse_scan_pce_without_modulator as Delta_PCE as main focus, use framework such as https://github.com/sjtu-sai-agents/ML-Master to perform exploratory data analysis (EDA) on QSPR. Genretate all reports to /share/yhm/test/AutoML_EDA


reports should be a complete science paper style report, including but not limited to:
figures, tables, figures captions, main tex and SI

you can clone git repo ML-Master and use its framework to perform the EDA.
