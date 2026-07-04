import pythia8
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D
import fastjet as fj  # FastJet for jet clustering
from scipy.spatial.distance import pdist, squareform

# Initialize Pythia
pythia = pythia8.Pythia()

# Configure Pythia for proton-proton collisions
pythia.readString("Beams:idA = 2212")  # Proton
pythia.readString("Beams:idB = 2212")  # Proton
pythia.readString("Beams:eCM = 14000.")  # Center-of-mass energy (14 TeV for LHC)
pythia.readString("HardQCD:all = on")  # Enable hard QCD processes
pythia.readString("PhaseSpace:pTHatMin = 20.")  # Minimum pT for hard scattering

# Initialize Pythia
if not pythia.init():
    raise RuntimeError("Pythia initialization failed.")

# Generate events
n_events = 1  # Number of events to process

for event_idx in range(n_events):
    if not pythia.next():
        continue

    # Collect final state particles for this event
    particles = []
    for particle in pythia.event:
        if particle.isFinal():  # Select final state particles
            particles.append([particle.px(), particle.py(), particle.pz(), particle.e()])

    particles = np.array(particles)

    # Skip events with too few particles
    if len(particles) < 2:
        print(f"Event {event_idx} has too few particles for clustering.")
        continue

    # ============================================
    # Jet Clustering with FastJet (anti-kt algorithm)
    # ============================================
    # Convert particles to FastJet format
    fj_particles = [fj.PseudoJet(px, py, pz, e) for px, py, pz, e in particles]

    # Define the jet definition (anti-kt algorithm, R=0.4)
    jet_def = fj.JetDefinition(fj.antikt_algorithm, 0.4)

    # Run the clustering
    cluster_sequence = fj.ClusterSequence(fj_particles, jet_def)
    fastjet_jets = cluster_sequence.inclusive_jets()

    # ============================================
    # Jet Clustering with DBSCAN
    # ============================================
    # Use (px, py, pz) for 3D clustering
    particle_data = particles[:, :3]

    # Standardize the data
    scaler = StandardScaler()
    particle_data_scaled = scaler.fit_transform(particle_data)

    # Apply DBSCAN clustering
    dbscan = DBSCAN(eps=0.3, min_samples=5)
    labels = dbscan.fit_predict(particle_data_scaled)

    # Extract DBSCAN jets
    dbscan_jets = []
    for k in set(labels):
        if k == -1:
            continue  # Skip noise
        jet_momentum = np.sum(particle_data[labels == k], axis=0)
        dbscan_jets.append(fj.PseudoJet(jet_momentum[0], jet_momentum[1], jet_momentum[2], np.linalg.norm(jet_momentum)))

    # ============================================
    # Compare Jets from FastJet and DBSCAN
    # ============================================
    print(f"\nEvent {event_idx}:")
    print(f"FastJet Jets: {len(fastjet_jets)}")
    print(f"DBSCAN Jets: {len(dbscan_jets)}")

    # Match jets based on Delta R
    def delta_r(jet1, jet2):
        deta = jet1.eta() - jet2.eta()
        dphi = np.arctan2(np.sin(jet1.phi() - jet2.phi()), np.cos(jet1.phi() - jet2.phi()))
        return np.sqrt(deta**2 + dphi**2)

    matched_jets = []
    for i, dbscan_jet in enumerate(dbscan_jets):
        for j, fastjet_jet in enumerate(fastjet_jets):
            if delta_r(dbscan_jet, fastjet_jet) < 0.4:  # Match within Delta R < 0.4
                matched_jets.append((i, j))
                print(f"Matched DBSCAN Jet {i} with FastJet Jet {j}")
                print(f"DBSCAN Jet: pT = {dbscan_jet.perp()}, eta = {dbscan_jet.eta()}, phi = {dbscan_jet.phi()}")
                print(f"FastJet Jet: pT = {fastjet_jet.perp()}, eta = {fastjet_jet.eta()}, phi = {fastjet_jet.phi()}")

    # ============================================
    # Plot Jets from FastJet and DBSCAN
    # ============================================
    fig = plt.figure(figsize=(14, 6))

    # Plot FastJet Jets
    ax1 = fig.add_subplot(121, projection='3d')
    for jet in fastjet_jets:
        ax1.scatter(jet.px(), jet.py(), jet.pz(), c='r', marker='o', s=100, label='FastJet Jet' if jet == fastjet_jets[0] else "")
    ax1.set_title(f'FastJet Jets (Event {event_idx})')
    ax1.set_xlabel('px')
    ax1.set_ylabel('py')
    ax1.set_zlabel('pz')

    # Plot DBSCAN Jets
    ax2 = fig.add_subplot(122, projection='3d')
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = (labels == k)
        xyz = particle_data[class_member_mask]
        ax2.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], 'o', c=[tuple(col)], marker='o', edgecolor='k', s=50,
                    label=f'DBSCAN Jet {k}' if k != -1 else "Noise")
    ax2.set_title(f'DBSCAN Jets (Event {event_idx})')
    ax2.set_xlabel('px')
    ax2.set_ylabel('py')
    ax2.set_zlabel('pz')

    plt.legend()
    plt.tight_layout()
    plt.show()