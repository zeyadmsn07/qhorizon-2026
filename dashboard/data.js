window.QH_DATA = {
 "meta": {
  "n_signals": 17898,
  "pulsar_pct": 9.16,
  "noise_pct": 90.84,
  "eval_seeds": "100..109",
  "n_seeds": 10,
  "n_train": 500,
  "pqk_circuits": 3500,
  "fqk_circuits": 1624750,
  "circuit_ratio": 464,
  "weights": 41
 },
 "htru2": [
  {
   "name": "Dummy",
   "label": "Dummy",
   "family": "baseline",
   "mean": 0.0916666666666666,
   "std": 0.0,
   "n": 10,
   "collapse": 10,
   "mcc": 0.0,
   "f1": 0.0,
   "params": 0.0,
   "sec": 0.00016655999934297002
  },
  {
   "name": "LogReg",
   "label": "LogReg",
   "family": "classical",
   "mean": 0.9183974674820308,
   "std": 0.016902500121260572,
   "n": 10,
   "collapse": 0,
   "mcc": 0.8618527944384837,
   "f1": 0.8737493861359168,
   "params": 9.0,
   "sec": 0.00572421999822832
  },
  {
   "name": "Q1-VQC (Bipartite_SignLinear_41p)",
   "label": "Q1-VQC (41p)",
   "family": "quantum",
   "mean": 0.9102010729249269,
   "std": 0.002494394461642948,
   "n": 10,
   "collapse": 0,
   "mcc": 0.8383711385202632,
   "f1": 0.8523583589503019,
   "params": 41.0,
   "sec": 11.359711150001385
  },
  {
   "name": "Q1-VQC (Ring L=2, Fragile v1)",
   "label": "Q1-VQC Ring v1",
   "family": "quantum",
   "mean": 0.8860303292491712,
   "std": 0.0061850312780867,
   "n": 10,
   "collapse": 0,
   "mcc": 0.8223346537937157,
   "f1": 0.8372544681744308,
   "params": 34.0,
   "sec": 17.969795350005732
  },
  {
   "name": "Q2-QuantumKernel (8q)",
   "label": "Q2-QuantumKernel (8q)",
   "family": "quantum",
   "mean": 0.9116410717087418,
   "std": 0.005680670887909939,
   "n": 10,
   "collapse": 0,
   "mcc": 0.8360002905792486,
   "f1": 0.8507569809220279,
   "params": 0.0,
   "sec": 0.26953651999647266
  },
  {
   "name": "RBF-SVM",
   "label": "RBF-SVM",
   "family": "classical",
   "mean": 0.9226714928348386,
   "std": 0.0036276798936304344,
   "n": 10,
   "collapse": 0,
   "mcc": 0.8579360771035656,
   "f1": 0.8701758142122182,
   "params": 0.0,
   "sec": 0.01157980001007666
  },
  {
   "name": "RandomForest",
   "label": "RandomForest",
   "family": "classical",
   "mean": 0.9015768591401075,
   "std": 0.008941836945371755,
   "n": 10,
   "collapse": 0,
   "mcc": 0.8358080466805594,
   "f1": 0.8493844487832469,
   "params": 0.0,
   "sec": 0.04455802000302353
  },
  {
   "name": "Tiny-MLP (41p)",
   "label": "Tiny-MLP (41p)",
   "family": "classical",
   "mean": 0.9246405775660074,
   "std": 0.007864689860155948,
   "n": 10,
   "collapse": 0,
   "mcc": 0.8632497926080953,
   "f1": 0.8753094535056484,
   "params": 41.0,
   "sec": 0.3222714799921959
  }
 ],
 "shell": [
  {
   "name": "LogReg",
   "label": "LogReg",
   "family": "classical",
   "mean": 0.1165288543825906,
   "lo": 0.1119137714728293,
   "hi": 0.1273819775094936
  },
  {
   "name": "RBF-SVM",
   "label": "RBF-SVM",
   "family": "classical",
   "mean": 0.8694737440522549,
   "lo": 0.839811277039375,
   "hi": 0.8973861396985295
  },
  {
   "name": "Tiny-MLP (41p)",
   "label": "Tiny-MLP (41p)",
   "family": "classical",
   "mean": 0.4075476213607133,
   "lo": 0.3425686497060466,
   "hi": 0.4784052519148316
  },
  {
   "name": "Q2-QuantumKernel (8q)",
   "label": "Q2-QuantumKernel (8q)",
   "family": "quantum",
   "mean": 0.883242143679148,
   "lo": 0.8512272638685985,
   "hi": 0.9103949596279788
  },
  {
   "name": "Q1-VQC (Headline)",
   "label": "Q1-VQC (41p)",
   "family": "quantum",
   "mean": 0.6602222915448149,
   "lo": 0.6075235896598464,
   "hi": 0.7096851738606456
  }
 ],
 "shell_by_n": {
  "100": [
   {
    "name": "LogReg",
    "label": "LogReg",
    "family": "classical",
    "mean": 0.1144388834098671,
    "lo": 0.1108347772533138,
    "hi": 0.1229436473648551
   },
   {
    "name": "RBF-SVM",
    "label": "RBF-SVM",
    "family": "classical",
    "mean": 0.4918800163950386,
    "lo": 0.4467403216911585,
    "hi": 0.540951932335745
   },
   {
    "name": "Tiny-MLP (41p)",
    "label": "Tiny-MLP (41p)",
    "family": "classical",
    "mean": 0.1124582138900641,
    "lo": 0.1089700855250783,
    "hi": 0.1201526248050908
   },
   {
    "name": "Q2-QuantumKernel (8q)",
    "label": "Q2-QuantumKernel (8q)",
    "family": "quantum",
    "mean": 0.5115260505587611,
    "lo": 0.4518744316684556,
    "hi": 0.5747446150360735
   },
   {
    "name": "Q1-VQC (Headline)",
    "label": "Q1-VQC (41p)",
    "family": "quantum",
    "mean": 0.1306849571577659,
    "lo": 0.1077557974244257,
    "hi": 0.1678218457035585
   }
  ],
  "200": [
   {
    "name": "LogReg",
    "label": "LogReg",
    "family": "classical",
    "mean": 0.116496475419072,
    "lo": 0.1119193734171956,
    "hi": 0.1258473696664616
   },
   {
    "name": "RBF-SVM",
    "label": "RBF-SVM",
    "family": "classical",
    "mean": 0.6712511944086516,
    "lo": 0.613853368799423,
    "hi": 0.7286100146090904
   },
   {
    "name": "Tiny-MLP (41p)",
    "label": "Tiny-MLP (41p)",
    "family": "classical",
    "mean": 0.2001487461435125,
    "lo": 0.1603139792276323,
    "hi": 0.2529287267209278
   },
   {
    "name": "Q2-QuantumKernel (8q)",
    "label": "Q2-QuantumKernel (8q)",
    "family": "quantum",
    "mean": 0.7461536580181245,
    "lo": 0.69362991467246,
    "hi": 0.793807352068437
   },
   {
    "name": "Q1-VQC (Headline)",
    "label": "Q1-VQC (41p)",
    "family": "quantum",
    "mean": 0.3127239363902898,
    "lo": 0.1891256055290061,
    "hi": 0.4284269644129483
   }
  ],
  "500": [
   {
    "name": "LogReg",
    "label": "LogReg",
    "family": "classical",
    "mean": 0.1165288543825906,
    "lo": 0.1119137714728293,
    "hi": 0.1273819775094936
   },
   {
    "name": "RBF-SVM",
    "label": "RBF-SVM",
    "family": "classical",
    "mean": 0.8694737440522549,
    "lo": 0.839811277039375,
    "hi": 0.8973861396985295
   },
   {
    "name": "Tiny-MLP (41p)",
    "label": "Tiny-MLP (41p)",
    "family": "classical",
    "mean": 0.4075476213607133,
    "lo": 0.3425686497060466,
    "hi": 0.4784052519148316
   },
   {
    "name": "Q2-QuantumKernel (8q)",
    "label": "Q2-QuantumKernel (8q)",
    "family": "quantum",
    "mean": 0.883242143679148,
    "lo": 0.8512272638685985,
    "hi": 0.9103949596279788
   },
   {
    "name": "Q1-VQC (Headline)",
    "label": "Q1-VQC (41p)",
    "family": "quantum",
    "mean": 0.6602222915448149,
    "lo": 0.6075235896598464,
    "hi": 0.7096851738606456
   }
  ]
 },
 "gold_a": [
  {
   "name": "VQC_bipartite_L1",
   "label": "VQC_bipartite_L1",
   "family": "quantum",
   "mean": 0.8782282073472565,
   "std": 0.009934242168315361,
   "n": 10,
   "collapse": 0,
   "mcc": 0.8161543170298062,
   "f1": 0.8294914648044717,
   "params": 18.0,
   "sec": 7.098334909995901
  },
  {
   "name": "VQC_bipartite_L2",
   "label": "VQC_bipartite_L2",
   "family": "quantum",
   "mean": 0.898051164817686,
   "std": 0.005730272105380575,
   "n": 10,
   "collapse": 0,
   "mcc": 0.833426276837276,
   "f1": 0.8475121348225982,
   "params": 34.0,
   "sec": 11.564234990000841
  },
  {
   "name": "VQC_bipartite_L4",
   "label": "VQC_bipartite_L4",
   "family": "quantum",
   "mean": 0.8989686250489051,
   "std": 0.011319132919555051,
   "n": 10,
   "collapse": 0,
   "mcc": 0.8282927990122542,
   "f1": 0.8430170577696645,
   "params": 66.0,
   "sec": 19.318130209998344
  },
  {
   "name": "VQC_none_L1",
   "label": "VQC_none_L1",
   "family": "quantum",
   "mean": 0.8861753187038348,
   "std": 0.00821015514758768,
   "n": 10,
   "collapse": 0,
   "mcc": 0.8181680571614989,
   "f1": 0.8325466444590159,
   "params": 18.0,
   "sec": 6.5996625700063305
  },
  {
   "name": "VQC_none_L2",
   "label": "VQC_none_L2",
   "family": "quantum",
   "mean": 0.8870612583747916,
   "std": 0.009762007186802727,
   "n": 10,
   "collapse": 0,
   "mcc": 0.8227314369937625,
   "f1": 0.8366868388041026,
   "params": 34.0,
   "sec": 10.57453569000354
  },
  {
   "name": "VQC_none_L4",
   "label": "VQC_none_L4",
   "family": "quantum",
   "mean": 0.8886219622890923,
   "std": 0.00860726007664401,
   "n": 10,
   "collapse": 0,
   "mcc": 0.818317403357159,
   "f1": 0.8329857190643661,
   "params": 66.0,
   "sec": 18.721595919996616
  },
  {
   "name": "VQC_ring_L1",
   "label": "VQC_ring_L1",
   "family": "quantum",
   "mean": 0.8643649038427549,
   "std": 0.0035045499606069595,
   "n": 10,
   "collapse": 0,
   "mcc": 0.8047573521727834,
   "f1": 0.8218204662922005,
   "params": 18.0,
   "sec": 7.347240540001076
  },
  {
   "name": "VQC_ring_L2",
   "label": "VQC_ring_L2",
   "family": "quantum",
   "mean": 0.8860303292491712,
   "std": 0.0061850312780867,
   "n": 10,
   "collapse": 0,
   "mcc": 0.8223346537937157,
   "f1": 0.8372544681744308,
   "params": 34.0,
   "sec": 17.969795350005732
  },
  {
   "name": "VQC_ring_L4",
   "label": "VQC_ring_L4",
   "family": "quantum",
   "mean": 0.8965830193979729,
   "std": 0.005962665427918133,
   "n": 10,
   "collapse": 0,
   "mcc": 0.8288538120014144,
   "f1": 0.8426151198636329,
   "params": 66.0,
   "sec": 21.782336859995848
  }
 ],
 "hardware": [
  {
   "model": "Q1-VQC (8q Bipartite Sign+Linear, 41p)",
   "qubits": 8,
   "depth": 7,
   "total_gates": 41,
   "two_qubit_gates": 8,
   "trainable_params": 41,
   "qpu_circuits_train_N500": "500/epoch (O(N))",
   "qpu_circuits_test_3000": "3,000 (O(N_test))",
   "shots_per_eval": "500-2,000 shots",
   "simulability": "<=8q (dim<=256), CPU-simulable"
  },
  {
   "model": "Q1-VQC (8q Ring L=2, 34p)",
   "qubits": 8,
   "depth": 21,
   "total_gates": 49,
   "two_qubit_gates": 16,
   "trainable_params": 34,
   "qpu_circuits_train_N500": "500/epoch (O(N))",
   "qpu_circuits_test_3000": "3,000 (O(N_test))",
   "shots_per_eval": "500-2,000 shots",
   "simulability": "<=8q (dim<=256), CPU-simulable"
  },
  {
   "model": "Q1-VQC (4q Amplitude+Norm L=4, 34p)",
   "qubits": 4,
   "depth": 25,
   "total_gates": 50,
   "two_qubit_gates": 16,
   "trainable_params": 34,
   "qpu_circuits_train_N500": "500/epoch (O(N))",
   "qpu_circuits_test_3000": "3,000 (O(N_test))",
   "shots_per_eval": "500-2,000 shots",
   "simulability": "<=8q (dim<=256), CPU-simulable"
  },
  {
   "model": "Q1-VQC (3q Amplitude L=5, 32p)",
   "qubits": 3,
   "depth": 26,
   "total_gates": 46,
   "two_qubit_gates": 15,
   "trainable_params": 32,
   "qpu_circuits_train_N500": "500/epoch (O(N))",
   "qpu_circuits_test_3000": "3,000 (O(N_test))",
   "shots_per_eval": "500-2,000 shots",
   "simulability": "<=8q (dim<=256), CPU-simulable"
  },
  {
   "model": "Q2-QuantumKernel (8q ZZ d=1, Inversion Test)",
   "qubits": 8,
   "depth": 52,
   "total_gates": 80,
   "two_qubit_gates": 32,
   "trainable_params": 0,
   "qpu_circuits_train_N500": "124,750 pairs (N*(N-1)/2 = O(N^2))",
   "qpu_circuits_test_3000": "1,500,000 pairs (N*N_test)",
   "shots_per_eval": "~10,000 shots/pair for +-0.01 K_ij",
   "simulability": "8q (dim=256), state-vector shortcut in sim"
  },
  {
   "model": "RBF-SVM (Classical Reference)",
   "qubits": 0,
   "depth": 0,
   "total_gates": 0,
   "two_qubit_gates": 0,
   "trainable_params": 0,
   "qpu_circuits_train_N500": "0 QPU runs",
   "qpu_circuits_test_3000": "0 QPU runs",
   "shots_per_eval": "Exact float64",
   "simulability": "Classical baseline (no quantum advantage claimed)"
  }
 ],
 "gold_b": [
  {
   "name": "3q_Amplitude_L5",
   "label": "3q_Amplitude_L5",
   "family": "quantum",
   "mean": 0.605716978444791,
   "std": 0.043291419088457614,
   "n": 10,
   "collapse": 10,
   "mcc": 0.6795118367773662,
   "f1": 0.7085738772778591,
   "params": 32.0,
   "sec": 3.151933210005518
  },
  {
   "name": "4q_AmplitudePlusNorm_L4",
   "label": "4q_AmplitudePlusNorm_L4",
   "family": "quantum",
   "mean": 0.8507296656654628,
   "std": 0.018964175015631343,
   "n": 10,
   "collapse": 0,
   "mcc": 0.8055171754313541,
   "f1": 0.8193923933656946,
   "params": 34.0,
   "sec": 3.76921399999992
  },
  {
   "name": "8q_Angle_L2",
   "label": "8q_Angle_L2",
   "family": "quantum",
   "mean": 0.8860303292491712,
   "std": 0.0061850312780867,
   "n": 10,
   "collapse": 0,
   "mcc": 0.8223346537937157,
   "f1": 0.8372544681744308,
   "params": 34.0,
   "sec": 17.969795350005732
  },
  {
   "name": "8q_Angle_Reupload_L2",
   "label": "8q_Angle_Reupload_L2",
   "family": "quantum",
   "mean": 0.8725700523645349,
   "std": 0.020590784243641868,
   "n": 10,
   "collapse": 0,
   "mcc": 0.8132779230367433,
   "f1": 0.828498507223066,
   "params": 34.0,
   "sec": 13.653448080000818
  }
 ],
 "gold_c": {
  "Q1-VQC (Bipartite_SignLinear_41p)": {
   "label": "Q1-VQC (41p)",
   "family": "quantum",
   "points": [
    {
     "n": 100,
     "mean": 0.8926208198767676,
     "std": 0.014455086313748991
    },
    {
     "n": 200,
     "mean": 0.9094613602156123,
     "std": 0.004736266192659614
    },
    {
     "n": 500,
     "mean": 0.9102010729249269,
     "std": 0.002494394461642948
    },
    {
     "n": 1000,
     "mean": 0.916193440820042,
     "std": 0.0034973546407308337
    }
   ]
  },
  "Q2-QuantumKernel (8q)": {
   "label": "Q2-QuantumKernel (8q)",
   "family": "quantum",
   "points": [
    {
     "n": 100,
     "mean": 0.8547571717917638,
     "std": 0.05608573886969586
    },
    {
     "n": 200,
     "mean": 0.8963746340800773,
     "std": 0.015402947247403992
    },
    {
     "n": 500,
     "mean": 0.9116410717087418,
     "std": 0.005680670887909939
    },
    {
     "n": 1000,
     "mean": 0.9270873835103574,
     "std": 0.006742598509482354
    }
   ]
  },
  "RBF-SVM": {
   "label": "RBF-SVM",
   "family": "classical",
   "points": [
    {
     "n": 100,
     "mean": 0.9007562522319578,
     "std": 0.013140666602528718
    },
    {
     "n": 200,
     "mean": 0.9134938684105733,
     "std": 0.00836607233934489
    },
    {
     "n": 500,
     "mean": 0.9226714928348386,
     "std": 0.0036276798936304344
    },
    {
     "n": 1000,
     "mean": 0.9293497336955421,
     "std": 0.002752534512443136
    }
   ]
  },
  "Tiny-MLP (41p)": {
   "label": "Tiny-MLP (41p)",
   "family": "classical",
   "points": [
    {
     "n": 100,
     "mean": 0.9040908029148902,
     "std": 0.013325641196854357
    },
    {
     "n": 200,
     "mean": 0.9175418128093764,
     "std": 0.00772438535352075
    },
    {
     "n": 500,
     "mean": 0.9246405775660074,
     "std": 0.007864689860155948
    },
    {
     "n": 1000,
     "mean": 0.933113127006151,
     "std": 0.0032425198507444455
    }
   ]
  }
 },
 "gold_d": [
  {
   "shots": "100",
   "label": "100 shots",
   "mean": 0.9033966737378586,
   "std": 0.0037281774266211196,
   "n": 10
  },
  {
   "shots": "500",
   "label": "500 shots",
   "mean": 0.9091159623794844,
   "std": 0.0020979685965599564,
   "n": 10
  },
  {
   "shots": "2000",
   "label": "2,000 shots",
   "mean": 0.9094792945472197,
   "std": 0.0020966899216417856,
   "n": 10
  },
  {
   "shots": "analytic",
   "label": "Exact (analytic)",
   "mean": 0.9102009573089698,
   "std": 0.0024942497357457643,
   "n": 10
  }
 ],
 "pqk": [
  {
   "alpha": 0.25,
   "pqk": 0.9233468410330146,
   "pqk_std": 0.0031748050952955,
   "fqk": 0.9166148590273744,
   "fqk_std": 0.0056206784848758
  },
  {
   "alpha": 0.5,
   "pqk": 0.914891259588166,
   "pqk_std": 0.0074546514229348,
   "fqk": 0.8847825387565408,
   "fqk_std": 0.0170677492104352
  },
  {
   "alpha": 1.0,
   "pqk": 0.8701586793214473,
   "pqk_std": 0.0156284496833488,
   "fqk": 0.8377775782005044,
   "fqk_std": 0.0325661438302101
  },
  {
   "alpha": 2.0,
   "pqk": 0.7752522865256328,
   "pqk_std": 0.0203407984352609,
   "fqk": 0.643558926705811,
   "fqk_std": 0.060866671304509
  },
  {
   "alpha": 4.0,
   "pqk": 0.6926041616110005,
   "pqk_std": 0.0275310427451608,
   "fqk": 0.5071059943823986,
   "fqk_std": 0.0517077005220537
  }
 ],
 "hybrid": [
  {
   "n": 100,
   "vqc": 0.8926208198767676,
   "mlp": 0.9040908029148902,
   "blend": 0.9125102280946268,
   "rank_blend": 0.9061996457684488
  },
  {
   "n": 200,
   "vqc": 0.9094613602156124,
   "mlp": 0.9175418128093764,
   "blend": 0.9209382811598996,
   "rank_blend": 0.9186293419287712
  },
  {
   "n": 500,
   "vqc": 0.9102010729249268,
   "mlp": 0.9246405775660074,
   "blend": 0.9223208318023802,
   "rank_blend": 0.921444059012256
  },
  {
   "n": 1000,
   "vqc": 0.9161934408200418,
   "mlp": 0.933113127006151,
   "blend": 0.9285039853360674,
   "rank_blend": 0.9295834141919916
  }
 ],
 "verdicts": [
  {
   "id": "H1a",
   "comparison": "Tiny-MLP (41p) vs. Q2-QuantumKernel (8q)",
   "delta": 0.012999505857265748,
   "lo": 0.004721954707496369,
   "hi": 0.02170023616616057,
   "p": 0.00390625,
   "verdict": "Supported"
  },
  {
   "id": "H1b",
   "comparison": "Tiny-MLP (41p) vs. Q1-VQC (Bipartite_SignLinear_41p)",
   "delta": 0.014439504641080592,
   "lo": 0.0059895457701099955,
   "hi": 0.023151751058042326,
   "p": 0.001953125,
   "verdict": "Supported"
  },
  {
   "id": "H1c",
   "comparison": "Q2-QuantumKernel (8q) vs. RandomForest",
   "delta": 0.010064212568634223,
   "lo": 0.0027454452276530466,
   "hi": 0.018901743649244237,
   "p": 0.001953125,
   "verdict": "Supported"
  },
  {
   "id": "H2a",
   "comparison": "VQC_bipartite_L2 vs. VQC_none_L2",
   "delta": 0.010989906442894482,
   "lo": -0.0005537246026741481,
   "hi": 0.022721983182218505,
   "p": 0.037109375,
   "verdict": "Inconclusive"
  },
  {
   "id": "H2b",
   "comparison": "VQC_bipartite_L2 vs. VQC_ring_L2",
   "delta": 0.012020835568514732,
   "lo": 0.0024775735705827875,
   "hi": 0.022517568984090575,
   "p": 0.001953125,
   "verdict": "Supported"
  },
  {
   "id": "H3",
   "comparison": "4q_AmplitudePlusNorm_L4 vs. 3q_Amplitude_L5",
   "delta": 0.24501268722067185,
   "lo": 0.19086864218922708,
   "hi": 0.2884683498564909,
   "p": 0.001953125,
   "verdict": "Supported"
  },
  {
   "id": "H4a",
   "comparison": "N=100 Q2-QuantumKernel vs. RBF-SVM (tied)",
   "delta": -0.04599908044019396,
   "lo": -0.08467513221728981,
   "hi": -0.014188421310061795,
   "p": 0.009765625,
   "verdict": "Contradicted"
  },
  {
   "id": "H4b_2000",
   "comparison": "Shots=2000 vs. Analytic (tied)",
   "delta": -0.0007216627617500859,
   "lo": -0.001522041480759817,
   "hi": 7.542152536586782e-06,
   "p": 0.005859375,
   "verdict": "Practically tied"
  },
  {
   "id": "H4b_500",
   "comparison": "Shots=500 vs. Analytic (tied)",
   "delta": -0.0010849949294853833,
   "lo": -0.002768973690824652,
   "hi": 0.0004797847420174083,
   "p": 0.16015625,
   "verdict": "Practically tied"
  },
  {
   "id": "H4b_100",
   "comparison": "Shots=100 vs. Analytic (tied)",
   "delta": -0.006804283571111225,
   "lo": -0.011189002527313622,
   "hi": -0.0030605940354479066,
   "p": 0.001953125,
   "verdict": "Inconclusive"
  }
 ]
};
