#!/usr/bin/env bash
# Extract individual neuron plots (016, 027, 072, 176) from the seed-43 analysis
# into ~/mental_navigation/selected_neurons/{target}/{version}/

set -euo pipefail

NEURONS=(016 027 072 176)
TARGETS=(hidden base_hidden)

MODEL_S43="random_vector_double_ctrnn_first_reg_dist_nodetach_rev_norm5_abs_N6L384I384Step64A0.9lr0.001F256S43Ep2000_CTSubpositiveE256_gNs384Np400D1Sig0.0_11_12_13"
ANALYSIS_BASE="$(dirname "$0")/../results/REV_ABS/analyze_two_rnn/dynamics/${MODEL_S43}/001999"
DEST_BASE="${HOME}/mental_navigation/selected_neurons"

for target in "${TARGETS[@]}"; do
    SRC="${ANALYSIS_BASE}/visual__0_${target}"
    DEST="${DEST_BASE}/${target}"

    SHIFTED_BASE="${SRC}/indiv_ignore_last_shiftedSig2Truncate2"
    STRETCHED_BASE="${SRC}/indiv_ignore_last_stretchedSig2Truncate2"

    # Create destination directories
    for opt in option1 option2 option3; do
        mkdir -p "${DEST}/shifted_${opt}"
        for version in onset offset best; do
            mkdir -p "${DEST}/stretched_${opt}_${version}"
        done
    done

    for n in "${NEURONS[@]}"; do
        # shifted — three alignment options
        for opt in option1 option2 option3; do
            src_f="${SHIFTED_BASE}/${opt}/firing_rate_shifted_${n}.pdf"
            [ -f "$src_f" ] && cp "$src_f" "${DEST}/shifted_${opt}/"
        done

        # stretched — three alignment options × three anchor versions
        for opt in option1 option2 option3; do
            for version in onset offset best; do
                src_f="${STRETCHED_BASE}/${opt}/${version}/firing_rate_stretched_${n}.pdf"
                [ -f "$src_f" ] && cp "$src_f" "${DEST}/stretched_${opt}_${version}/"
            done
        done
    done

    echo "Done: ${target}"
done

echo "All neurons extracted to ${DEST_BASE}"
