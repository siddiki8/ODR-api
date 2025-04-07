# Introduction

Grid-scale energy storage systems (ESS) are increasingly recognized as critical components for modernizing electricity grids and facilitating the transition towards a sustainable energy future. The proliferation of intermittent renewable energy sources like wind and solar necessitates robust storage solutions to balance supply and demand, enhance grid stability and reliability, and maximize the utilization of clean energy. Stationary energy storage can provide numerous grid services, including frequency regulation, peak shaving, load leveling, ancillary services, and support for integrating variable renewable generation. This report provides a comparative analysis of four prominent grid-scale energy storage technologies: lithium-ion (Li-ion) batteries, pumped hydro storage (PHS), compressed air energy storage (CAES), and flow batteries (specifically vanadium redox flow batteries, VRFB, unless otherwise specified), evaluating their technical performance, economic viability, environmental impact, and grid integration suitability based on scientific literature and industry reports.

*   **Lithium-ion (Li-ion) Batteries:** These electrochemical batteries store energy via the movement of lithium ions between electrodes. Dominant chemistries for grid applications include Lithium Iron Phosphate (LFP) and Nickel Manganese Cobalt Oxide (NMC). They are known for high energy density, high efficiency, and fast response times, benefiting from cost reductions driven by the electric vehicle market.
*   **Pumped Hydro Storage (PHS):** The most mature and widely deployed grid-scale storage technology, PHS uses gravitational potential energy. Water is pumped from a lower reservoir to an upper reservoir during periods of low demand/excess generation and released through turbines to generate electricity during peak demand. Its deployment is geographically constrained.
*   **Compressed Air Energy Storage (CAES):** This technology stores energy by compressing air into underground caverns (e.g., salt domes) or artificial reservoirs. When electricity is needed, the compressed air is released, often heated (diabatic CAES, typically using natural gas), and expanded through turbines. Adiabatic and isothermal CAES concepts aim to improve efficiency and eliminate fossil fuel use but are less mature. Like PHS, it often faces geographical limitations.
*   **Flow Batteries:** These electrochemical systems store energy in liquid electrolytes held in external tanks. Energy capacity is scaled by increasing electrolyte volume, while power is scaled by increasing the size/number of electrochemical cell stacks. Vanadium redox flow batteries (VRFBs) are a leading chemistry known for long cycle life and independent scaling of power and energy.

# Technical Performance

Technical performance characteristics, including efficiency, lifespan, and energy density, vary significantly among the four storage technologies, influencing their suitability for different grid applications.

**Round-Trip Efficiency (RTE):** RTE measures the ratio of energy output during discharge to energy input during charge.
*   **Li-ion Batteries:** Exhibit the highest RTE among the compared technologies, typically ranging from 85% to 98% (DC-DC) or around 84% (AC-AC) for LFP and NMC chemistries. High efficiency minimizes energy losses during charge-discharge cycles.
*   **Pumped Hydro Storage (PHS):** Offers good efficiency, generally between 70% and 85%, with modern variable-speed plants achieving the higher end of this range.
*   **Compressed Air Energy Storage (CAES):** Efficiency varies significantly by design. Older diabatic plants (using natural gas for heating during expansion) range from 42% (Huntorf) to 54% (McIntosh). More modern diabatic designs may reach up to 60%. Storelectric claims 68-70% for their adiabatic CAES concept. Adiabatic and isothermal CAES aim for higher efficiencies but are less commercially mature. Hydrogen storage systems (sometimes compared with CAES for long duration) have lower RTE, around 31% AC-AC.
*   **Flow Batteries (VRFB):** Generally have lower RTE compared to Li-ion, typically ranging from 60% to 85% (AC-AC around 65-71% reported). Losses occur in pumping electrolytes and electrochemical conversions.

**Lifespan:** Lifespan can be measured in calendar years or charge/discharge cycles.
*   **Li-ion Batteries:** Have a limited lifespan compared to mechanical storage, typically 10–20 years calendar life and 1,500–10,000 cycles depending on chemistry (LFP generally offers higher cycle life than NMC), depth of discharge (DOD), temperature, and operating conditions. Capacity degrades over time, often defined as end-of-life when capacity reaches 60-80% of the initial rating. Frequent cycling can shorten lifespan, requiring augmentation or replacement during a project's economic life.
*   **Pumped Hydro Storage (PHS):** Offers very long operational lifetimes, often 40–60 years or more, with potential extension up to 100 years with proper maintenance and component replacement. Cycle life is typically not a limiting factor.
*   **Compressed Air Energy Storage (CAES):** Similar to PHS, CAES plants have long expected lifespans, around 30-60 years for mechanical components and potentially over 100 years for underground caverns. Cycle life limitations are generally not a primary concern.
*   **Flow Batteries (VRFB):** Offer potentially very high cycle lives (10,000+ cycles, sometimes cited as unlimited for the electrolyte itself). The stack components may have a calendar life of 10–20 years, requiring replacement, but the electrolyte retains value and can be reused.

**Energy Density:** Refers to the amount of energy stored per unit volume or mass. This is less critical for stationary grid storage than for mobile applications but impacts footprint.
*   **Li-ion Batteries:** Have the highest energy density among the electrochemical options considered (typically 200–400 Wh/L), leading to a relatively smaller physical footprint for a given energy capacity compared to flow or lead-acid batteries.
*   **Pumped Hydro Storage (PHS):** Very low energy density (0.2–2 Wh/L), requiring large geographical areas and significant elevation differences for reservoirs. Footprint is substantial.
*   **Compressed Air Energy Storage (CAES):** Low energy density (2–6 Wh/L), heavily dependent on the volume and pressure of the storage cavern or vessel. Requires large underground formations or significant artificial reservoirs.
*   **Flow Batteries (VRFB):** Have lower energy density (20–70 Wh/L) compared to Li-ion batteries, requiring larger tanks for electrolyte storage and thus a larger footprint for the energy component, although the power component (stack) footprint is separate.

**Summary Table of Typical Technical Performance:**

| Feature             | Li-ion (LFP/NMC) | Pumped Hydro (PHS) | Compressed Air (CAES - Diabatic) | Flow Battery (VRFB) |
| :------------------ | :--------------- | :----------------- | :------------------------------- | :------------------ |
| **RTE (AC-AC)**     | ~84%             | 70–85%             | 42–60%                           | 65–71%              |
| **Calendar Life**   | 10–20 years      | 40–60+ years       | 30–60 years                      | 10–20 years (stack) |
| **Cycle Life**      | 1,500–10,000+    | Very High          | Very High                        | 10,000+ (unlimited electrolyte) |
| **Energy Density**  | High (200–400 Wh/L)| Very Low (0.2–2 Wh/L)| Low (2–6 Wh/L)                 | Low (20–70 Wh/L)    |
| **Response Time**   | Milliseconds     | Seconds to Minutes | Seconds to Minutes               | Milliseconds to Seconds |
| **Power/Energy Decoupling** | No | Yes                | Yes                              | Yes                 |

*(Note: Values are indicative ranges from sources and can vary based on specific design, chemistry, and operating conditions.)*

# Economic Viability

Economic viability is assessed through capital costs (upfront investment) and the Levelized Cost of Storage (LCOS), which represents the average cost per unit of discharged energy over the project's lifetime.

**Capital Costs:** Often broken down into power-related costs ($/kW) and energy-related costs ($/kWh).
*   **Li-ion Batteries:** Have seen significant cost reductions, driven by EV market scale. PNNL 2021 estimates for a 100 MW, 10-hour system are $356/kWh (LFP) and $405/kWh (NMC). Costs decrease with longer duration but less steeply than PHS or CAES because battery modules (energy component) are a major cost driver. Projections for 2030 show further decreases (e.g., LFP to ~$230-290/kWh for 4-10 hour systems).
*   **Pumped Hydro Storage (PHS):** High upfront capital costs, dominated by civil engineering for reservoirs and powerhouse construction. Costs are highly site-specific. PNNL 2021 estimate for 100 MW, 10-hour system is $263/kWh, decreasing significantly for longer durations ($143/kWh for 24 hours). Navigant 2019 estimates $170.3/kWh (average 2019-2028). Less potential for dramatic cost reduction due to technology maturity.
*   **Compressed Air Energy Storage (CAES):** Costs are highly dependent on geology (natural caverns are cheapest). Diabatic CAES is estimated by PNNL 2022 as the lowest cost option for durations ≥ 4 hours ($122/kWh for 100 MW, 10-hour system using caverns). Artificial storage significantly increases costs. Projections for 2030 show CAES remaining very competitive, especially at long durations ($18/kWh for 100 MW, 100-hour system using caverns).
*   **Flow Batteries (VRFB):** Capital costs are influenced by vanadium prices, though leasing models exist. PNNL 2021 estimate for a 100 MW, 10-hour system is $385/kWh. Costs decrease significantly with duration due to the low marginal cost of adding electrolyte tanks. Projections for 2030 show significant cost reductions, potentially becoming competitive with Li-ion at longer durations.

**Levelized Cost of Storage (LCOS):** Accounts for capital costs, O&M, efficiency, lifespan, replacement/augmentation costs, financing, etc.
*   **Li-ion Batteries:** PNNL 2022 estimates 2021 LCOS for 10 MW systems (≤10 hours) between $0.20–$0.40/kWh (LFP lowest, Lead-Acid highest). LCOS increases significantly beyond ~10 hours duration due to underutilization relative to cycle life limits or calendar life fade. Navigant 2019 estimated a higher LCOS for Li-ion ($285/MWh or $0.285/kWh) compared to PHS over a 40-year life, factoring in replacements. Projections for 2030 show decreasing LCOS, potentially making Li-ion competitive up to ~8 hours even at large scale (100-1000 MW).
*   **Pumped Hydro Storage (PHS):** Benefits from very long lifespan and low O&M, leading to competitive LCOS despite high initial capex. Navigant 2019 estimated $186/MWh ($0.186/kWh) over 40 years. PNNL 2022 shows PHS having the second-lowest LCOS ($0.11/kWh for 1000 MW, 10-hour) after CAES for large-scale, long-duration applications in 2021. OUP 2025 reports PHS life cycle cost per MWh ($66.5) as roughly half that of LIB in China.
*   **Compressed Air Energy Storage (CAES):** Generally offers the lowest LCOS for durations ≥ 4 hours at large scale, particularly when using low-cost natural caverns. PNNL 2022 estimates $0.10/kWh for a 1,000 MW, 10-hour system in 2021. Lower sensitivity to electricity purchase price compared to batteries.
*   **Flow Batteries (VRFB):** LCOS is influenced by RTE, stack replacement costs, and initial capex. PNNL 2022 shows VRFB LCOS becoming competitive with Li-ion around 6-10 hour durations in 2021 ($0.19/kWh for 1000 MW, 10-hour). Unlimited electrolyte life is an advantage.

**Economic Comparison Summary:**
*   For short-to-medium durations (≤ 4-8 hours), Li-ion batteries (especially LFP) are becoming increasingly cost-competitive due to falling capex, although LCOS reflects replacement needs.
*   For large-scale, long-duration applications (≥ 8-10 hours), PHS and CAES (with favorable geology) currently offer the lowest capital costs ($/kWh) and LCOS.
*   Flow batteries show promise for medium-to-long durations, with costs strongly dependent on duration and vanadium prices. Their LCOS can be competitive with Li-ion in the 6-10+ hour range.
*   Future cost projections suggest Li-ion costs will continue to fall significantly, potentially making them competitive at longer durations by 2030, though PHS and CAES are expected to retain advantages for very long durations (24-100+ hours).

# Environmental Impact

The environmental impacts of grid-scale storage technologies encompass their full lifecycle, including raw material extraction, manufacturing, operation (including charging source), and end-of-life management (recycling and disposal).

**Lifecycle Emissions (GHG/Carbon Footprint):**
*   **Overall:** Operational emissions depend heavily on the source of electricity used for charging. Using renewable energy significantly lowers the use-phase emissions for all technologies. Infrastructure (manufacturing, construction, decommissioning) impacts vary by technology.
*   **Li-ion Batteries:** Manufacturing is energy-intensive and a significant source of lifecycle emissions. LCA studies show LIB currently having favorable environmental performance compared to PHES in some contexts (OUP 2025), but this depends heavily on grid mix and system boundaries. Projections suggest LIBs could achieve carbon neutrality by 2030 with decarbonization measures (OUP 2025). Recycling processes also have associated emissions.
*   **Pumped Hydro Storage (PHS):** Construction (concrete, steel, land use change) dominates infrastructure emissions. Operational emissions depend on charging electricity source and potential methane emissions from reservoirs (though lower for closed-loop systems). LCA studies suggest PHS has the lowest life cycle GWP compared to other storage options when charged with renewables (ACS 2023). OUP 2025 projects PHS reaching carbon neutrality by 2040 in China.
*   **Compressed Air Energy Storage (CAES):** Diabatic CAES using natural gas during discharge generates direct GHG emissions. Adiabatic and isothermal designs aim to eliminate this. Construction emissions are significant, particularly for artificial storage. LCA results vary; some studies show CAES emissions lower than PHS when coupled with fossil generation due to fuel displacement, but higher when coupled with renewables (ACS 2023 - Denholm).
*   **Flow Batteries (VRFB):** Manufacturing of components (membranes, stacks) and production/purification of vanadium electrolyte contribute to lifecycle emissions. Operational emissions depend on charging source and auxiliary power needs (pumps). LCA comparisons vary; some show VRFB having higher impacts than LIB (ScienceDirect 2025), while others suggest potential advantages depending on specific chemistries and system designs.

**Material Sourcing and End-of-Life:**
*   **Li-ion Batteries:** Raise significant concerns regarding the sourcing of lithium and cobalt (often mined in conflict zones with associated human rights and environmental issues) and graphite. Recycling infrastructure is still developing, and recovering high-purity materials economically remains challenging. Improper disposal poses risks due to hazardous materials. LFP chemistries avoid cobalt, mitigating some ethical/supply chain risks.
*   **Pumped Hydro Storage (PHS):** Primarily uses common construction materials (concrete, steel). Land use and ecosystem impacts (habitat disruption, altered hydrology for open-loop) are major concerns during construction and operation. Decommissioning costs are often considered small on a present value basis due to long lifespans.
*   **Compressed Air Energy Storage (CAES):** Relies heavily on suitable geology (salt caverns, aquifers, depleted mines) or requires construction of artificial reservoirs (steel tanks). Surface footprint is comparable to a power station. Air is the storage medium, posing minimal sourcing issues. Potential environmental impacts relate to site construction and subsurface integrity.
*   **Flow Batteries (VRFB):** Vanadium is the key material. While relatively abundant, sourcing and processing have environmental impacts. Electrolyte is corrosive and requires careful handling, but vanadium itself can be fully recycled/reused almost indefinitely, representing a significant circular economy advantage over Li-ion. Stack materials require end-of-life management.

**Comparative Environmental Aspects:**
*   **Land Use:** PHS and CAES (especially with artificial storage) generally have larger physical footprints or specific geographical requirements compared to battery systems.
*   **Resource Depletion/Toxicity:** Li-ion faces challenges with critical materials (Li, Co). Flow batteries rely on vanadium. PHS/CAES use more common materials but involve significant construction impacts. Toxicity concerns exist for electrolytes in Li-ion and flow batteries, and for lead in lead-acid batteries.
*   **Recycling/Circularity:** VRFB offers high recyclability potential for vanadium electrolyte. Li-ion recycling is improving but faces economic and technical hurdles. PHS/CAES components are largely conventional materials but involve large-scale decommissioning.
*   **Safety:** Li-ion batteries pose fire risks (thermal runaway) requiring sophisticated safety systems. Flow batteries generally use non-flammable electrolytes but may involve corrosive materials. PHS/CAES involve risks associated with large civil structures and high pressures.

# Grid Integration Suitability

The suitability of each technology for grid integration depends on its technical characteristics, economic profile, and ability to provide specific grid services.
*   **Lithium-ion Batteries:**
    *   **Strengths:** Fast response time (milliseconds) makes them ideal for frequency regulation and ancillary services. Modular design allows flexible scaling and installation. High energy density leads to smaller footprints. Declining costs enhance competitiveness.
    *   **Challenges:** Limited cycle life requires careful management and potential replacement/augmentation for long-duration or high-cycling applications. Relatively shorter discharge durations (typically ≤ 4-8 hours economically) limit suitability for bulk, long-duration storage compared to PHS/CAES. Safety concerns require robust thermal management and fire suppression systems. High upfront cost for very large capacities.
    *   **Applications:** Frequency regulation, peak shaving, renewable energy integration (smoothing short-term variability), transmission/distribution deferral, power quality.
*   **Pumped Hydro Storage (PHS):**
    *   **Strengths:** Proven, mature technology with very long lifespan and high reliability. Ideal for large-scale, long-duration bulk energy storage (hours to days). Provides grid inertia. Low operating costs.
    *   **Challenges:** High upfront capital cost and long development timelines (permitting, construction). Strict geographical and topographical requirements limit potential sites. Significant land use and potential environmental/social impacts associated with reservoir construction. Slower response time compared to batteries.
    *   **Applications:** Bulk energy shifting (arbitrage), long-duration storage, grid inertia, black start capability, ancillary services (with variable speed units).
*   **Compressed Air Energy Storage (CAES):**
    *   **Strengths:** Suitable for large-scale, long-duration storage (hours to days). Potentially very low LCOS if suitable geology is available. Long lifespan. Provides grid inertia similar to PHS.
    *   **Challenges:** Highly dependent on specific geological formations (salt caverns, aquifers) for cost-effectiveness; artificial storage significantly increases cost. Diabatic CAES relies on fossil fuels (typically natural gas) and has associated emissions and lower RTE. Adiabatic/isothermal CAES are less mature. Longer response times than batteries.
    *   **Applications:** Bulk energy shifting, long-duration storage, renewable energy integration, grid inertia.
*   **Flow Batteries (VRFB):**
    *   **Strengths:** Decoupling of power and energy allows independent scaling; energy capacity can be increased cost-effectively by adding more electrolyte. Very long cycle life (>10,000 cycles) and potential for electrolyte reuse reduces lifetime costs and environmental impact. Good safety profile (non-flammable electrolyte).
    *   **Challenges:** Lower RTE and energy density compared to Li-ion. Higher upfront costs than Li-ion for shorter durations. Stack replacement may be needed during project life. Vanadium cost volatility (though leasing helps). Larger footprint for energy component.
    *   **Applications:** Medium-to-long duration storage (e.g., 4-12+ hours), renewable energy shifting, grid stabilization, applications where high cycle life is critical.

**Integration Considerations:**
*   **Flexibility:** Batteries and flow batteries offer greater siting flexibility than PHS/CAES. Modularity allows incremental deployment.
*   **Duration Needs:** Li-ion excels at short durations (<4-8 hours). Flow batteries are competitive at medium-to-long durations. PHS and CAES dominate very long duration, bulk storage needs.
*   **Grid Services:** Fast response of batteries is valuable for frequency regulation. Inertia from PHS/CAES is beneficial for grid stability.
*   **Regulatory/Market Frameworks:** Policies are evolving to properly value the services provided by different storage types and remove barriers to entry, which is crucial for deployment.

# Conclusion

Grid-scale energy storage is essential for enabling a high penetration of renewable energy sources and ensuring grid reliability. Lithium-ion batteries, pumped hydro storage, compressed air energy storage, and flow batteries each present distinct advantages and disadvantages across technical, economic, environmental, and grid integration metrics.

*   **Lithium-ion Batteries (LFP/NMC):** Offer high efficiency, high energy density, and rapid response, making them suitable for frequency regulation and short-to-medium duration applications (up to ~8 hours). Costs are falling rapidly due to EV market synergies. However, their limited cycle/calendar life necessitates careful management and potential replacement, impacting long-term LCOS, and material sourcing (especially cobalt for NMC) and recycling pose environmental challenges. Safety management is critical.
*   **Pumped Hydro Storage (PHS):** Remains the dominant technology for large-scale, long-duration bulk storage due to its maturity, very long lifespan, high reliability, and provision of grid inertia. While capital costs are high and site-specific, its LCOS is very competitive for long durations. Major limitations are geographical constraints and long development times.
*   **Compressed Air Energy Storage (CAES):** Provides a cost-effective solution for large-scale, long-duration storage, potentially offering the lowest LCOS where suitable geology exists. Like PHS, it offers long life and grid inertia. Key challenges include site dependency, the lower efficiency and emissions of current diabatic designs, and the relative immaturity of adiabatic/isothermal alternatives.
*   **Flow Batteries (VRFB):** Are well-suited for medium-to-long duration storage (4-12+ hours) due to their decoupled power/energy scaling and very high cycle life. The potential for electrolyte reuse enhances sustainability. Current drawbacks include lower RTE and energy density compared to Li-ion, and higher costs for shorter durations. Vanadium price volatility is a factor, though leasing mitigates this.

**Suitability Summary:**
No single technology is optimal for all applications. The choice depends on specific grid needs, duration requirements, scale, location, and economic context.
*   **Short Duration / High Power / Fast Response (e.g., Frequency Regulation):** Li-ion batteries are generally preferred.
*   **Bulk Energy Shifting / Long Duration (≥10 hours):** PHS and CAES (with favorable geology) are typically the most economical options currently.
*   **Medium-to-Long Duration (4-12+ hours) / High Cycling:** Flow batteries offer a compelling alternative, particularly where long cycle life is paramount and footprint is less constrained than for Li-ion.

Future energy grids will likely rely on a portfolio of storage technologies, leveraging the strengths of each to provide a full range of services across different timescales and scales. Continued research and development focusing on cost reduction, performance improvement (especially efficiency and lifespan for batteries), safety, and sustainable lifecycle management (materials sourcing, recycling) will be crucial for all technologies to fulfill their potential in supporting a decarbonized and reliable energy system.
Sources Consulted:
1. [[PDF] 2022 Grid Energy Storage Technology Cost and Performance ...](https://www.pnnl.gov/sites/default/files/media/file/ESGC%20Cost%20Performance%20Report%202022%20PNNL-33283.pdf)
2. [Techno-economic benefits of grid-scale energy storage in future energy systems](https://www.sciencedirect.com/science/article/pii/S2352484720302444)
3. [Rechargeable batteries for grid scale energy storage](https://pubs.acs.org/doi/abs/10.1021/acs.chemrev.2c00289)
4. [[PDF] Comparing the Costs of Long Duration Energy Storage Technologies](https://www.slenergystorage.com/documents/20190626_Long_Duration%20Storage_Costs.pdf)
5. [Eco-economic comparison of batteries and pumped-hydro systems ...](https://www.sciencedirect.com/science/article/pii/S0196890424004680)
6. [Life cycle environmental and economic impacts of various energy ...](https://academic.oup.com/ieam/advance-article/doi/10.1093/inteam/vjaf035/8046021?searchresult=1)
7. [Applications of lithium-ion batteries in grid-scale energy storage systems](https://link.springer.com/article/10.1007/s12209-020-00236-w)
8. [Integrated life cycle assessment and techno-economic analysis of ...](https://www.sciencedirect.com/science/article/abs/pii/S2352550925000107)
9. [Utility-Scale Energy Storage - C3 Controls](https://www.c3controls.com/white-paper/utility-scale-energy-storage/?srsltid=AfmBOoo8_dim0vqMFkA_MT7WHgOw-ofdeWu_qpdE1zzV59A65RYeYsGB)
10. [Economic and financial appraisal of novel large-scale energy storage technologies](https://www.sciencedirect.com/science/article/pii/S0360544220320612)
11. [A comprehensive review of stationary energy storage devices for large scale renewable energy sources grid integration](https://www.sciencedirect.com/science/article/pii/S1364032122001368)
12. [Life Cycle Assessment of Closed-Loop Pumped Storage ...](https://pubs.acs.org/doi/10.1021/acs.est.2c09189)
13. [On-grid batteries for large-scale energy storage: Challenges and opportunities for policy and technology](https://www.cambridge.org/core/journals/mrs-energy-and-sustainability/article/ongrid-batteries-for-largescale-energy-storage-challenges-and-opportunities-for-policy-and-technology/3671E7C0E8F8B570FDA6C8321E5DD441)
14. [Utility-Scale Energy Storage - C3 Controls](https://www.c3controls.com/white-paper/utility-scale-energy-storage/?srsltid=AfmBOop88JbnyiNm9cruS8AbFsoeDL7L2FgK4g5Zt5NX4FbQ_kNu2SNg)
15. [CAES or Batteries in the Energy Transition? - Storelectric](https://storelectric.com/caes-or-batteries-in-the-energy-transition/)
16. [Overview of lithium-ion grid-scale energy storage systems](https://link.springer.com/article/10.1007/s40518-017-0086-0)
17. [Techno-economic comparison of utility-scale compressed air and electro-chemical storage systems](https://www.mdpi.com/1996-1073/15/18/6644)
18. [Lithium-Ion Energy Storage Cost vs. Pumped Hydro Or Flow Battery ...](https://cleantechnica.com/2020/04/25/how-long-can-lithium-ion-energy-storage-actually-last/)
19. [Key Challenges for grid‐scale lithium‐ion battery energy storage](https://onlinelibrary.wiley.com/doi/abs/10.1002/aenm.202202197)
20. [Key Challenges for Grid‐Scale Lithium‐Ion Battery Energy Storage](https://onlinelibrary.wiley.com/doi/full/10.1002/aenm.202202197)
21. [Life Cycle Assessment and Costing of Large-Scale Battery Energy Storage Integration in Lombok's Power Grid](https://www.mdpi.com/2313-0105/10/8/295)
22. [Overview of current development in compressed air energy storage technology](https://www.sciencedirect.com/science/article/pii/S1876610214034547)
23. [A review of pumped hydro storage systems](https://www.mdpi.com/1996-1073/16/11/4516)
24. [Pumped storage hydropower for sustainable and low-carbon electricity grids in pacific rim economies](https://www.mdpi.com/1996-1073/15/9/3139)
25. [Comprehensive review of compressed air energy storage (CAES) technologies](https://www.mdpi.com/2673-7264/3/1/8)
26. [On the economics of storage for electricity: Current state and future ...](https://wires.onlinelibrary.wiley.com/doi/10.1002/wene.431)
27. [What Are the Potential Environmental Impacts of Large-Scale ...](https://sustainability-directory.com/question/what-are-the-potential-environmental-impacts-of-large-scale-energy-storage-systems/)
28. [Fact Sheet | Energy Storage (2019) | White Papers | EESI](https://www.eesi.org/papers/view/energy-storage-2019)
29. [Grid-scale battery costs: the economics? - Thunder Said Energy](https://thundersaidenergy.com/downloads/battery-storage-costs-the-economics/)
30. [Techno-economic analysis of bulk-scale compressed air energy storage in power system decarbonisation](https://www.sciencedirect.com/science/article/pii/S0306261920315208)
31. [Environmental benefit-detriment thresholds for flow battery energy storage systems: A case study in California](https://www.sciencedirect.com/science/article/pii/S0306261921007613)
32. [Techno‐economic study of compressed air energy storage systems for the grid integration of wind power](https://onlinelibrary.wiley.com/doi/abs/10.1002/er.3840)
33. [News Release: NREL Analysis Reveals Benefits of Hydropower for ...](https://www.nrel.gov/news/press/2023/news-release-nrel-analysis-reveals-benefits-of-hydropower-for-grid-scale-energy-storage.html)
34. [Techno-economic assessment and grid impact of Thermally-Integrated Pumped Thermal Energy Storage (TI-PTES) systems coupled with photovoltaic plants …](https://www.sciencedirect.com/science/article/pii/S2352152X23032978)