# Compiled workflow diagrams

Generated with `python -m scripts.export_graph`; both diagrams come from `get_graph().draw_mermaid()`. The model adapter is used only to compile structure. This export performs no inference and is not execution evidence.

## Coordinator

```mermaid
---
config:
  flowchart:
    curve: linear
---
graph TD;
	__start__([<p>__start__</p>]):::first
	input_validation(input_validation)
	planner(planner)
	supervisor_plan_review(supervisor_plan_review)
	approval(approval)
	researcher(researcher)
	analyst(analyst)
	supervisor_evidence_review(supervisor_evidence_review)
	replan(replan)
	report_generator(report_generator)
	validator(validator)
	semantic_validator(semantic_validator)
	supervisor_final_review(supervisor_final_review)
	finalize(finalize)
	recovery(recovery)
	__end__([<p>__end__</p>]):::last
	__start__ --> input_validation;
	analyst -.-> recovery;
	analyst -.-> supervisor_evidence_review;
	approval -.-> recovery;
	approval -.-> researcher;
	input_validation -.-> planner;
	input_validation -.-> recovery;
	planner -.-> recovery;
	planner -.-> supervisor_plan_review;
	replan -.-> recovery;
	replan -.-> researcher;
	report_generator -.-> recovery;
	report_generator -.-> validator;
	researcher -.-> analyst;
	researcher -.-> recovery;
	semantic_validator -.-> recovery;
	semantic_validator -.-> supervisor_final_review;
	supervisor_evidence_review -.-> recovery;
	supervisor_evidence_review -.-> replan;
	supervisor_evidence_review -.-> report_generator;
	supervisor_final_review -.-> finalize;
	supervisor_final_review -.-> recovery;
	supervisor_final_review -.-> replan;
	supervisor_final_review -.-> report_generator;
	supervisor_plan_review -.-> approval;
	supervisor_plan_review -.-> planner;
	supervisor_plan_review -.-> recovery;
	validator -.-> recovery;
	validator -.-> semantic_validator;
	finalize --> __end__;
	recovery --> __end__;
	researcher -.-> researcher;
	classDef default fill:#f2f0ff,line-height:1.2
	classDef first fill-opacity:0
	classDef last fill:#bfb6fc

```

## Researcher tool loop

```mermaid
---
config:
  flowchart:
    curve: linear
---
graph TD;
	__start__([<p>__start__</p>]):::first
	model(model)
	execute_tools(execute_tools)
	__end__([<p>__end__</p>]):::last
	__start__ --> model;
	execute_tools --> model;
	model -.-> __end__;
	model -.-> execute_tools;
	model -.-> model;
	classDef default fill:#f2f0ff,line-height:1.2
	classDef first fill-opacity:0
	classDef last fill:#bfb6fc

```
