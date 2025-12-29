# Optimizing Multi-Layer Transshipment Networks

This project formulates and solves a multi-layer transshipment problem using Linear Programming to minimize total transportation cost across supply, transshipment, and demand nodes.

## Problem Overview
In a transshipment network, flow can move from supply nodes to demand nodes either directly or through intermediate transshipment nodes.  
The goal is to decide how much to ship on each arc while satisfying supply/demand and capacity constraints at minimum cost.

## Approach
- Modeled the network as a directed graph with multiple layers (Supply → Transshipment → Demand)
- Decision variables: flow on each arc
- Objective: minimize total transportation cost
- Constraints:
  - Supply limits at supply nodes
  - Demand fulfillment at demand nodes
  - Flow conservation at transshipment nodes
  - Capacity bounds on arcs (if provided)
- Solved the LP model with **PuLP** and analyzed scenario changes (e.g., capacities/costs) to compare outcomes.

## Tech Stack
- **Python**
- **PuLP** (Linear Programming)
- **NetworkX** (network representation)
- **Matplotlib** (visualization)

## How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/OmerOsman34/Optimizing-Multi-Layer-Transshipment-Networks.git
