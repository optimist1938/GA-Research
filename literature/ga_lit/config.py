"""Configuration of the literature pipeline: what to search for and how to recognise a hit.

Every regular expression here is case-insensitive. Three vocabularies drive the triage score:

* ``GA_TERMS``   - words that indicate Clifford / geometric algebra content.
* ``ML_TERMS``   - words that indicate a machine-learning or neural-network context.
* ``EXCLUDE_TERMS`` - name collisions that are *not* geometric algebra (quantum "Clifford gates",
  the Clifford torus, "multivector" document retrieval, rotor machines, ...).

A candidate paper is worth a human look when it matches at least one GA term and one ML term and no
exclusion term. Everything that scores is written to the ``inbox``; nothing is accepted automatically.
"""

from __future__ import annotations

import re

# --------------------------------------------------------------------------------------------------
# Vocabularies
# --------------------------------------------------------------------------------------------------

GA_TERMS = re.compile(
    r"geometric algebra|clifford algebra|clifford-algebra|clifford group|clifford neural|clifford layer|"
    r"clifford convolution|clifford-steerable|clifford steerable|clifford-valued|clifford valued|"
    r"clifford (network|transformer|attention|equivariant|graph|sheaf|kolmogorov|fourier|neuron|embedding|"
    r"latent|product|kernel|module|block|representation|feature|variational|neural)|"
    r"multivector|multi-vector|\bgatr\b|l-gatr|geometric product|\bversor|bivector|spacetime algebra|"
    r"conformal geometric|projective geometric|\bpin\(\d|\bspin\(\d|\bcl\(\d|hypercomplex",
    re.I,
)

ML_TERMS = re.compile(
    r"neural|learning|network|transformer|deep|equivarian|attention|embedding|policy|diffusion|surrogate|"
    r"classif|regress|predict|encoder|autoencoder|language model|\bllm\b|reinforcement|generative|"
    r"segmentation|detection|recognition|hopfield|synchroni[sz]ation",
    re.I,
)

EXCLUDE_TERMS = re.compile(
    r"clifford (circuit|gate|stabilizer|hierarchy|unitar|operation|simulat|random|twirl|tableau|element|"
    r"disentangl|\+t|invariants and three-copy)|stabilizer|qubit|pauli string|magic state|clifford torus|"
    r"clifford-klein|clifford parallel|clifford index|clifford modules?|clifford bundle|spin\^c|"
    r"dirac operator|clifford analysis|monogenic|hecke-clifford|clifford-fischer|"
    r"multi-?vector (retrieval|rerank|embedding|index|attack|model predictive|predictive control)|"
    r"\brotor (blade|machine|bar|fault|dynamics|aircraft)|rotorcraft|kicked rotor|tilt-rotor|quadrotor|"
    r"wind turbine|induction motor",
    re.I,
)

# Keyword -> topic suggestions used to pre-fill ``topics`` of inbox items. The human reviewer decides.
TOPIC_HINTS = {
    "survey": re.compile(r"\bsurvey\b|\breview\b|overview|position paper", re.I),
    "hep": re.compile(r"jet|collider|lhc|lorentz|quark|hadron|amplitude|particle physics|top-quark|calorimeter", re.I),
    "scientific-ml": re.compile(r"interatomic|molecul|catalys|pde\b|fluid|navier|weather|climate|sea level|material|lattice qcd", re.I),
    "vision": re.compile(r"image|vision|hyperspectral|segmentation|registration|video|cifar|imagenet", re.I),
    "graphs": re.compile(r"graph neural|message passing|sheaf|multimodal.attributed|knowledge graph", re.I),
    "robotics": re.compile(r"robot|manipulat|self-driving|autonomous driving|odometry|kinematic|grasp|slam", re.I),
    "wireless": re.compile(r"wireless|channel prediction|nerf|radiance field|gaussian splatting", re.I),
    "sequence-nlp": re.compile(r"time series|forecast|sequence model|language|semantic|recurren|\bllm\b", re.I),
    "clifford-valued-dynamics": re.compile(r"synchroni[sz]ation|stability|delay|fractional-order|hopfield|cohen.grossberg|\bbam\b|memristive", re.I),
    "hypercomplex": re.compile(r"hypercomplex|quaternion-valued|biquaternion|split-quaternion", re.I),
    "theory": re.compile(r"lie group|identifiab|expressiv|universal approximation|scalarization|theorem|convexity", re.I),
}

# --------------------------------------------------------------------------------------------------
# Source queries
# --------------------------------------------------------------------------------------------------

# arXiv API ``search_query`` fragments. The date window is appended by the fetcher.
ARXIV_QUERIES = [
    'all:"clifford algebra"', 'all:"geometric algebra"', 'all:clifford AND all:neural',
    'all:clifford AND all:equivariant', 'all:clifford AND all:learning', 'all:"clifford group"',
    'all:multivector', 'all:multivectors', 'all:"geometric algebra" AND all:transformer',
    'all:"clifford layers"', 'all:GATr', 'all:"projective geometric algebra"', 'all:"conformal geometric algebra"',
    'all:rotor AND all:neural', 'all:spinor AND all:neural', 'all:bivector AND all:learning',
    'all:"geometric product" AND all:neural', 'all:"geometric algebra" AND all:learning',
    'all:"geometric algebra" AND all:network', 'all:"clifford" AND all:transformer',
    'all:"clifford" AND all:graph AND all:neural', 'all:"clifford" AND all:"machine learning"',
    'all:"geometric algebra" AND all:"machine learning"', 'abs:clifford AND cat:cs.LG', 'abs:clifford AND cat:cs.CV',
    'abs:clifford AND cat:cs.RO', 'abs:clifford AND cat:stat.ML', 'abs:"geometric algebra"',
    'all:"clifford-valued"', 'all:"clifford valued"', 'all:"versor" AND all:neural',
    'all:"geometric algebra" AND all:equivariant', 'all:"Lorentz" AND all:"geometric algebra"',
    'all:"Cl(" AND all:neural', 'all:hypercomplex AND all:neural', 'all:"spacetime algebra" AND all:learning',
]

# OpenReview full-text search terms (``api2.openreview.net/notes/search``).
OPENREVIEW_TERMS = [
    "Clifford", "Clifford algebra", "geometric algebra", "multivector", "GATr", "geometric product",
    "Clifford group", "versor", "bivector", "Clifford-valued",
]

# Crossref ``query.bibliographic`` terms.
CROSSREF_TERMS = [
    "Clifford algebra neural network", "geometric algebra neural network", "Clifford equivariant neural",
    "geometric algebra deep learning", "multivector neural network", "Clifford convolution",
    "geometric algebra transformer", "Clifford-valued neural networks", "geometric algebra machine learning",
    "Clifford algebra machine learning", "Clifford neural", "geometric algebra attention",
    "geometric algebra graph neural network", "Clifford algebra transformer", "geometric algebra learning",
    "hypercomplex neural network", "conformal geometric algebra learning", "projective geometric algebra learning",
]

# INSPIRE-HEP records whose citing papers we track (``refersto:recid:<n>``). The number is the INSPIRE
# control number of the paper named in the key.
INSPIRE_SEEDS = {
    "L-GATr (Spinner et al. 2024)": 2789600,
    "L-GATr for all of the LHC (Brehmer et al. 2024)": 2844989,
    "Clifford Group Equivariant Neural Networks (Ruhe et al. 2023)": 2661033,
    "Economical Jet Taggers (Petitjean et al. 2025)": 3094401,
    "Geometric algebra as the input language of collider foundation models (Abasov et al. 2026)": 3156709,
}
INSPIRE_TITLE_QUERIES = ['t "geometric algebra"', 't clifford and not t circuit and not t gate']

# Polite contact string sent in User-Agent / mailto parameters (Crossref asks for one).
CONTACT = "ga-research literature sweep (https://github.com/chagrygoris/GA-Research)"

# Seconds to wait between requests to the same API.
SLEEP = {"arxiv": 3.0, "openreview": 1.0, "crossref": 1.0, "inspire": 1.0}
