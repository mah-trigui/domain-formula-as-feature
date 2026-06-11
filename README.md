# Use the Domain Formula as a Feature

A machine learning project showing how a first-principles analytical estimate can be injected into a regression model as a structured prior rather than used as the final prediction.

## Project Website

[View Project Site](https://mah-trigui.github.io/domain-formula-as-feature/)

## Overview

This project predicts delivery time for a logistics platform in Nairobi using a hybrid analytical-ML approach.

The key idea was to build a baseline duration estimate from distance and rider historical speed, then feed that estimate into XGBoost as a feature. The model then learned the correction from contextual signals such as rider behavior, location, and operational timing.

## Core Modeling Pattern

Instead of choosing between:
- formula-based estimation
- pure machine learning

the project combines both:

- the formula captures the stable relationship
- the model captures the residual deviation

## Architecture

![Architecture](images/architecture.jpg)

<details>
<summary>📋 View detailed text-based architecture diagram</summary>

```text
                  ┌─────────────────────────────────────┐
                  │  Raw Delivery Context              │
                  │  - distance                        │
                  │  - rider history                   │
                  │  - pickup / arrival timestamps     │
                  │  - location coordinates            │
                  └────────────────┬────────────────────┘
                                   │
                                   ▼
               ┌────────────────────────────────────────────┐
               │ Analytical Baseline Construction           │
               │ expected_duration ≈ distance / rider_speed │
               └────────────────┬───────────────────────────┘
                                │
                                ▼
               ┌────────────────────────────────────────────┐
               │ Baseline-Derived Features                  │
               │ - dur_estim                                │
               │ - dur_estim_avg                            │
               │ - dur_nor                                  │
               │ - imp                                      │
               └────────────────┬───────────────────────────┘
                                │
          ┌─────────────────────┼──────────────────────────────┐
          ▼                     ▼                              ▼
┌──────────────────┐  ┌────────────────────┐      ┌────────────────────┐
│ Rider Priors     │  │ Geographic Context │      │ Operational Timing │
│ - speed_med      │  │ - pickup clusters  │      │ - plac_confir      │
│ - speed_avg      │  │ - dest clusters    │      │ - haversine        │
│ - best_rider     │  │                    │      │ - arriv_pick       │
└─────────┬────────┘  └──────────┬─────────┘      └──────────┬─────────┘
          └──────────────────────┴────────────────────────────┘
                                 │
                                 ▼
                    ┌────────────────────────────┐
                    │      XGBoost Regressor     │
                    │  learns residual structure │
                    └──────────────┬─────────────┘
                                   │
                                   ▼
                    ┌────────────────────────────┐
                    │ Final Delivery Time Pred.  │
                    │ formula prior + correction │
                    └────────────────────────────┘
