import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="TOPSIS avec Entropie", layout="wide", page_icon="📊")

# Titre principal
st.title("🎯 Méthode TOPSIS avec Entropie et Poids Hiérarchiques")
st.markdown("---")

# Initialisation de session_state
if 'decision_matrix' not in st.session_state:
    st.session_state.decision_matrix = None
if 'main_criteria' not in st.session_state:
    st.session_state.main_criteria = []

# Sidebar pour la configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    num_alternatives = st.number_input("Nombre d'alternatives", min_value=2, max_value=20, value=4)
    num_main_criteria = st.number_input("Nombre de critères principaux", min_value=1, max_value=10, value=3)
    
    st.markdown("---")
    st.info("📚 **Méthode TOPSIS**\n\nCombine les poids d'entropie (objectifs) et les poids subjectifs pour un classement optimal des alternatives.")

# Fonction pour calculer l'entropie
def calculate_entropy_weights(matrix):
    """Calcule les poids d'entropie pour chaque critère"""
    m, n = matrix.shape
    k = 1 / np.log(m)
    
    # Normalisation
    P = matrix / matrix.sum(axis=0)
    
    # Calcul de Z (éviter log(0))
    P_safe = np.where(P == 0, 1e-10, P)
    Z = P_safe / P_safe.sum(axis=0)
    
    # Calcul de l'entropie
    e = -k * np.sum(Z * np.log(Z), axis=0)
    
    # Calcul des poids
    w_entropy = (1 - e) / np.sum(1 - e)
    
    return w_entropy, e

# Fonction pour combiner les poids
def combine_weights(w_entropy, w_subjective):
    """Combine les poids d'entropie et subjectifs"""
    w_combined = (w_entropy * w_subjective) / np.sum(w_entropy * w_subjective)
    return w_combined

# Fonction TOPSIS complète
def topsis_analysis(matrix, weights, criteria_types):
    """
    Effectue l'analyse TOPSIS complète
    matrix: matrice de décision
    weights: poids combinés
    criteria_types: 'benefit' ou 'cost' pour chaque critère
    """
    m, n = matrix.shape
    
    # Étape 2: Normalisation
    P = matrix / matrix.sum(axis=0)
    
    # Étape 6: Application des poids
    U = P * weights
    
    # Étape 7: Solutions idéales
    A_plus = np.zeros(n)
    A_minus = np.zeros(n)
    
    for j in range(n):
        if criteria_types[j] == 'benefit':
            A_plus[j] = np.max(U[:, j])
            A_minus[j] = np.min(U[:, j])
        else:  # cost
            A_plus[j] = np.min(U[:, j])
            A_minus[j] = np.max(U[:, j])
    
    # Étape 8: Calcul des distances
    S_plus = np.sqrt(np.sum((U - A_plus)**2, axis=1))
    S_minus = np.sqrt(np.sum((U - A_minus)**2, axis=1))
    
    # Étape 9: Proximité relative
    C = S_minus / (S_plus + S_minus)
    
    # Étape 10: Classement
    ranking = np.argsort(-C) + 1
    
    return {
        'normalized_matrix': P,
        'weighted_matrix': U,
        'A_plus': A_plus,
        'A_minus': A_minus,
        'S_plus': S_plus,
        'S_minus': S_minus,
        'proximity': C,
        'ranking': ranking
    }

# Onglets principaux
tab1, tab2, tab3, tab4 = st.tabs(["📝 Critères", "📊 Matrice de Décision", "🧮 Calculs", "📈 Résultats"])

with tab1:
    st.header("Définition des Critères Principaux et Sous-critères")
    
    main_criteria_data = []
    
    for i in range(num_main_criteria):
        st.subheader(f"🔹 Critère Principal {i+1}")
        
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            main_name = st.text_input(f"Nom", value=f"Critère Principal {i+1}", key=f"main_{i}")
        with col2:
            main_weight = st.number_input(f"Poids subjectif", min_value=0.0, max_value=1.0, value=1/num_main_criteria, step=0.01, key=f"weight_{i}")
        with col3:
            num_sub = st.number_input(f"Nb sous-critères", min_value=1, max_value=10, value=2, key=f"numsub_{i}")
        
        sub_criteria = []
        for j in range(num_sub):
            col_a, col_b = st.columns([3, 1])
            with col_a:
                sub_name = st.text_input(f"  └─ Sous-critère {i+1}.{j+1}", value=f"Sous-critère {i+1}.{j+1}", key=f"sub_{i}_{j}")
            with col_b:
                sub_type = st.selectbox(f"Type", ["benefit", "cost"], key=f"type_{i}_{j}")
            
            sub_criteria.append({
                'name': sub_name,
                'type': sub_type
            })
        
        main_criteria_data.append({
            'name': main_name,
            'weight': main_weight,
            'sub_criteria': sub_criteria
        })
        
        st.markdown("---")
    
    st.session_state.main_criteria = main_criteria_data
    
    # Vérification des poids
    total_weight = sum([mc['weight'] for mc in main_criteria_data])
    if abs(total_weight - 1.0) > 0.01:
        st.warning(f"⚠️ La somme des poids subjectifs est {total_weight:.2f}. Elle devrait être égale à 1.0")
    else:
        st.success(f"✅ Somme des poids subjectifs = {total_weight:.2f}")

with tab2:
    st.header("Matrice de Décision")
    
    if st.session_state.main_criteria:
        # Créer la liste de tous les sous-critères
        all_sub_criteria = []
        criteria_types = []
        main_criteria_indices = []
        
        for mc_idx, mc in enumerate(st.session_state.main_criteria):
            for sc in mc['sub_criteria']:
                all_sub_criteria.append(f"{mc['name']}: {sc['name']}")
                criteria_types.append(sc['type'])
                main_criteria_indices.append(mc_idx)
        
        num_sub_criteria = len(all_sub_criteria)
        
        st.info(f"📋 Total: {num_alternatives} alternatives × {num_sub_criteria} sous-critères")
        
        # Créer le DataFrame pour la saisie
        if st.session_state.decision_matrix is None or st.session_state.decision_matrix.shape != (num_alternatives, num_sub_criteria):
            st.session_state.decision_matrix = pd.DataFrame(
                np.random.randint(1, 10, size=(num_alternatives, num_sub_criteria)),
                columns=all_sub_criteria,
                index=[f"Alternative {i+1}" for i in range(num_alternatives)]
            )
        
        st.markdown("### 📝 Saisir les valeurs de performance")
        edited_df = st.data_editor(
            st.session_state.decision_matrix,
            use_container_width=True,
            num_rows="fixed"
        )
        
        st.session_state.decision_matrix = edited_df
        
        # Afficher les types de critères
        st.markdown("### 🏷️ Types de critères")
        types_df = pd.DataFrame({
            'Sous-critère': all_sub_criteria,
            'Type': criteria_types,
            'Critère Principal': [st.session_state.main_criteria[idx]['name'] for idx in main_criteria_indices]
        })
        st.dataframe(types_df, use_container_width=True)

with tab3:
    st.header("Calculs Détaillés")
    
    if st.session_state.decision_matrix is not None and st.button("🚀 Lancer les calculs", type="primary"):
        
        matrix = st.session_state.decision_matrix.values
        
        # Calculer les poids d'entropie pour chaque sous-critère
        w_entropy, entropy_values = calculate_entropy_weights(matrix)
        
        # Préparer les poids subjectifs des sous-critères
        w_subjective = []
        for mc_idx, mc in enumerate(st.session_state.main_criteria):
            main_weight = mc['weight']
            num_sub = len(mc['sub_criteria'])
            # Répartir le poids principal également entre les sous-critères
            for _ in range(num_sub):
                w_subjective.append(main_weight / num_sub)
        
        w_subjective = np.array(w_subjective)
        
        # Normaliser les poids subjectifs
        w_subjective = w_subjective / w_subjective.sum()
        
        # Combiner les poids
        w_combined = combine_weights(w_entropy, w_subjective)
        
        # Extraire les types de critères
        criteria_types = []
        for mc in st.session_state.main_criteria:
            for sc in mc['sub_criteria']:
                criteria_types.append(sc['type'])
        
        # Analyse TOPSIS
        results = topsis_analysis(matrix, w_combined, criteria_types)
        
        st.session_state.results = {
            'w_entropy': w_entropy,
            'entropy_values': entropy_values,
            'w_subjective': w_subjective,
            'w_combined': w_combined,
            'topsis': results,
            'criteria_types': criteria_types
        }
        
        # Affichage des résultats intermédiaires
        st.success("✅ Calculs terminés avec succès!")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Nombre d'alternatives", num_alternatives)
        with col2:
            st.metric("Nombre de sous-critères", len(w_entropy))
        with col3:
            best_alt = np.argmax(results['proximity']) + 1
            st.metric("Meilleure alternative", f"Alternative {best_alt}")
        
        # Tableau des poids
        st.markdown("### ⚖️ Comparaison des Poids")
        weights_df = pd.DataFrame({
            'Sous-critère': st.session_state.decision_matrix.columns,
            'Entropie (Objectif)': w_entropy,
            'Subjectif': w_subjective,
            'Combiné': w_combined,
            'Type': criteria_types
        })
        st.dataframe(weights_df.style.format({
            'Entropie (Objectif)': '{:.4f}',
            'Subjectif': '{:.4f}',
            'Combiné': '{:.4f}'
        }), use_container_width=True)
        
        # Matrice normalisée
        st.markdown("### 📐 Matrice Normalisée")
        normalized_df = pd.DataFrame(
            results['normalized_matrix'],
            columns=st.session_state.decision_matrix.columns,
            index=st.session_state.decision_matrix.index
        )
        st.dataframe(normalized_df.style.format('{:.4f}'), use_container_width=True)
        
        # Matrice pondérée
        st.markdown("### 🎯 Matrice Pondérée (U)")
        weighted_df = pd.DataFrame(
            results['weighted_matrix'],
            columns=st.session_state.decision_matrix.columns,
            index=st.session_state.decision_matrix.index
        )
        st.dataframe(weighted_df.style.format('{:.4f}'), use_container_width=True)

with tab4:
    st.header("Résultats Finaux")
    
    if 'results' in st.session_state and st.session_state.results is not None:
        results = st.session_state.results
        topsis = results['topsis']
        
        # Tableau de classement
        st.markdown("### 🏆 Classement Final")
        
        ranking_df = pd.DataFrame({
            'Alternative': st.session_state.decision_matrix.index,
            'S+ (Distance PIS)': topsis['S_plus'],
            'S- (Distance NIS)': topsis['S_minus'],
            'Proximité Relative (Ci)': topsis['proximity'],
            'Rang': topsis['ranking']
        }).sort_values('Rang')
        
        st.dataframe(ranking_df.style.format({
            'S+ (Distance PIS)': '{:.4f}',
            'S- (Distance NIS)': '{:.4f}',
            'Proximité Relative (Ci)': '{:.4f}'
        }).background_gradient(subset=['Proximité Relative (Ci)'], cmap='RdYlGn'), use_container_width=True)
        
        # Graphiques
        col1, col2 = st.columns(2)
        
        with col1:
            # Graphique à barres des proximités
            fig1 = go.Figure(data=[
                go.Bar(
                    x=st.session_state.decision_matrix.index,
                    y=topsis['proximity'],
                    marker_color=topsis['proximity'],
                    marker_colorscale='RdYlGn',
                    text=[f"{v:.4f}" for v in topsis['proximity']],
                    textposition='outside'
                )
            ])
            fig1.update_layout(
                title="Proximité Relative par Alternative",
                xaxis_title="Alternative",
                yaxis_title="Proximité (Ci)",
                yaxis_range=[0, 1]
            )
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            # Comparaison des distances
            fig2 = go.Figure()
            fig2.add_trace(go.Bar(
                name='S+ (Distance PIS)',
                x=st.session_state.decision_matrix.index,
                y=topsis['S_plus'],
                marker_color='lightcoral'
            ))
            fig2.add_trace(go.Bar(
                name='S- (Distance NIS)',
                x=st.session_state.decision_matrix.index,
                y=topsis['S_minus'],
                marker_color='lightgreen'
            ))
            fig2.update_layout(
                title="Comparaison des Distances",
                xaxis_title="Alternative",
                yaxis_title="Distance",
                barmode='group'
            )
            st.plotly_chart(fig2, use_container_width=True)
        
        # Radar chart pour visualiser les performances
        st.markdown("### 🎯 Analyse des Performances")
        
        selected_alts = st.multiselect(
            "Sélectionner les alternatives à comparer",
            options=list(st.session_state.decision_matrix.index),
            default=list(st.session_state.decision_matrix.index)[:min(3, num_alternatives)]
        )
        
        if selected_alts:
            fig3 = go.Figure()
            
            for alt in selected_alts:
                idx = list(st.session_state.decision_matrix.index).index(alt)
                values = topsis['weighted_matrix'][idx].tolist()
                values.append(values[0])  # Fermer le radar
                
                criteria_names = list(st.session_state.decision_matrix.columns)
                criteria_names.append(criteria_names[0])
                
                fig3.add_trace(go.Scatterpolar(
                    r=values,
                    theta=criteria_names,
                    fill='toself',
                    name=alt
                ))
            
            fig3.update_layout(
                polar=dict(radialaxis=dict(visible=True)),
                showlegend=True,
                title="Comparaison des Performances Pondérées"
            )
            st.plotly_chart(fig3, use_container_width=True)
        
        # Téléchargement des résultats
        st.markdown("### 💾 Exporter les Résultats")
        
        col_a, col_b = st.columns(2)
        with col_a:
            csv = ranking_df.to_csv(index=False)
            st.download_button(
                label="📥 Télécharger le classement (CSV)",
                data=csv,
                file_name="topsis_ranking.csv",
                mime="text/csv"
            )
        
        with col_b:
            weights_csv = pd.DataFrame({
                'Critère': st.session_state.decision_matrix.columns,
                'Poids_Entropie': results['w_entropy'],
                'Poids_Subjectif': results['w_subjective'],
                'Poids_Combiné': results['w_combined']
            }).to_csv(index=False)
            st.download_button(
                label="📥 Télécharger les poids (CSV)",
                data=weights_csv,
                file_name="topsis_weights.csv",
                mime="text/csv"
            )
    
    else:
        st.info("👆 Veuillez d'abord lancer les calculs dans l'onglet 'Calculs'")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>📊 Application TOPSIS avec Entropie et Poids Hiérarchiques</p>
    <p>Méthode combinant poids objectifs (entropie) et subjectifs pour une prise de décision optimale</p>
</div>
""", unsafe_allow_html=True)