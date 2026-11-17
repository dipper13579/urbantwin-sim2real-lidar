# External release asset inventory

52 original artifacts: 48 checkpoints, two final predictions and two historical submission archives. Every file is below 2 GB; none belongs in Git. Machine-readable inventory: `provenance/release_assets.json`. Source fields describe the neutral WORK_ROOT checkpoint layout or ASSET_ROOT download filenames; original development locations are retained only in the private author handoff.

**Historical archives are hash references, not automatically cleared for redistribution:** they contain synthetic points and the original declaration, which may identify the signer. Before an owner-authorized upload, review dataset redistribution terms and replace the declaration if privacy requires it. A changed archive needs a new filename/hash; preserve the historical hash as a reference. Repack only after explicit publication authorization.

| Filename | Member / submission | Bytes | SHA-256 |
|---|---|---:|---|
| `lumpi__A_cp20__model.pth` | lumpi/A_cp20 | 102496009 | `bbae6a26f3e6ae447f877fc87a6ecb4ce35910b0f0d357ad1d7e86ef1536ed1c` |
| `lumpi__F_v3__model.pth` | lumpi/F_v3 | 104806744 | `18ecc8d18b1fde1c33d883b119ea7d14d4807ab1cd34f4f1f24ad813175a6ab8` |
| `lumpi__F_v8__model.pth` | lumpi/F_v8 | 104806745 | `4f0cbb2571a08e3f4128fe4fda9b8392309d4d304517e7636ac10f3a668f3a54` |
| `lumpi__F_v4__model.pth` | lumpi/F_v4 | 104803908 | `91ee96a767bddb32e7c77dd5d550ba434416f14277b64e16029f4e3513e28da1` |
| `lumpi__F_v7__model.pth` | lumpi/F_v7 | 104806745 | `0243a845f1e63bd78d17982428ba62c83d0b4e606340a828827c9b51259cec33` |
| `lumpi__F_v9__model.pth` | lumpi/F_v9 | 104803907 | `9d372c3f6ba303313e43b67b12ad5cee137ede7e4250d74423f4c04ccf3acb16` |
| `lumpi__F_v11__model.pth` | lumpi/F_v11 | 104806745 | `1ba74af437906ad0a4d7215b1fd39166e5c2753a98963dfdf1cb8bd9bb3447ac` |
| `lumpi__F_v16__model.pth` | lumpi/F_v16 | 115423585 | `7c0370ea3aa69adccf1991ccbe2efcd9860bd45fddd3e9561d4bb50fc9c6de6a` |
| `lumpi__F_v19__model.pth` | lumpi/F_v19 | 104803907 | `ca254543597e74885a20dc9f433d4b36086a13912edf778342dcaad9895a941a` |
| `lumpi__F_v22__model.pth` | lumpi/F_v22 | 104806745 | `ee6f68dce89f3729b4b546a512a87f103ae9a3d06bc11652103229c7def2e80a` |
| `lumpi__F_v23__model.pth` | lumpi/F_v23 | 104803907 | `c8142f153f6367b0e140dee830ffffd0f7a073f215cfaa74db02f0e8b5acf1f4` |
| `lumpi__F_v24__model.pth` | lumpi/F_v24 | 104806746 | `e8154f84e3410ad765fabcf1e5b36084fdd3cd04efe05b3a5a4e7518b01c98af` |
| `lumpi__F_v25__model.pth` | lumpi/F_v25 | 104806745 | `4d2626c30c038b499e41bf8098ede3fc710e142a1a071f0663a7d213f6fa4908` |
| `lumpi__F_v26__model.pth` | lumpi/F_v26 | 115420745 | `f70cb78bea408ff8c633e8525da20067c98a00e364bd5c3df173c6d45cbbce03` |
| `lumpi__F_v27__model.pth` | lumpi/F_v27 | 115420745 | `25ad92b0b83984364e9dd06d0db5ec54043946f945fe6a83091d4180ed78ee89` |
| `lumpi__F_v30__model.pth` | lumpi/F_v30 | 104806746 | `8de7b659b160e0df0ef003793cbdbf32c69ef0090ba6435f4d8db70267350d19` |
| `lumpi__F_v34__model.pth` | lumpi/F_v34 | 104803907 | `b45bfdaa0ea103dfca5b12fbe4aa203f87eada5195f5ed8fb6d0d629ad9cc6ea` |
| `lumpi__F_v35__model.pth` | lumpi/F_v35 | 115423583 | `45c769beeddc9b11f3c82a2f7c2a2722d207223815a5248b9960de8fce2f0f21` |
| `lumpi__F_v36__model.pth` | lumpi/F_v36 | 104803907 | `0f85628fba75e3d99dd866fbbd7f25a9199640d47c8dec3ff193519d0fed99cf` |
| `v2x__V25__model.pth` | v2x/V25 | 108477603 | `073d97c476e5fc5b35155de488473430fb65877cb8083eaff334062169e4c00f` |
| `v2x__V12__model.pth` | v2x/V12 | 108475575 | `11d08129fd7dca3c11cf04842d7528191587a3f7926c3bfc6c55c61c7a056d0b` |
| `v2x__V10__model.pth` | v2x/V10 | 108475575 | `6f1f4ac61073c921a3cc94f2585d7522c72f08f9cdb4531457ae407c3b8c29d9` |
| `v2x__V9__model.pth` | v2x/V9 | 108477603 | `f6d6cfeedd54325a1bbca0449af0676a9f455b817066f38ca72584cc88435090` |
| `v2x__V8__model.pth` | v2x/V8 | 108477603 | `56f75342b1e252eb483634594328342a3bcfdc1ff98af1222dec29046980e547` |
| `v2x__V7__model.pth` | v2x/V7 | 108475575 | `a0c51723c3c69c0b239a4d4996fa2c6d28154911044f9c203fc15ca34ab47908` |
| `v2x__V6__model.pth` | v2x/V6 | 108477603 | `fbaa79a3d3a0dfc4d63f857eddb381ffea4dbb27cfccd50dd0aa2db2b107aea7` |
| `v2x__V5__model.pth` | v2x/V5 | 108477603 | `2238827fea02f26228e852da9b3a3eacbbe577a065a117678f664a8f195f9f7b` |
| `v2x__V4__model.pth` | v2x/V4 | 108477602 | `931ef47b014efe6580ce94b28b88bd8be6089c0bca76524e8d53e6059c9d2127` |
| `v2x__V21__model.pth` | v2x/V21 | 108475575 | `1725fea6422de3959e5698890e9762cad42d7237866757ee645c1d277152aee5` |
| `v2x__V26__model.pth` | v2x/V26 | 108477603 | `dc8565986c8388f36c68a73294bdc6fcafe8bc0fc81faba195b113537c34fa8c` |
| `v2x__V22__model.pth` | v2x/V22 | 108477603 | `ec495077f8b49d096f9290179d3ab574940e3b1af8e8d43abe8ae3201b23814d` |
| `v2x__V27__model.pth` | v2x/V27 | 108477603 | `12670395ed171b0cf804f9f94b3b6d643b83be02f595e796087284618918cd64` |
| `v2x__V16__model.pth` | v2x/V16 | 108477603 | `543d3beff05ef8c90bc82b9e38d2ed0cd4116e7ffa06244212e8b8726fb454b4` |
| `v2x__V28__model.pth` | v2x/V28 | 108475575 | `769f7aa063a11eb767a7336bfe5e73d23a073aa529c6566d15f46ea5c69fdad1` |
| `v2x__V30__model.pth` | v2x/V30 | 108477603 | `cf07fdf6ee70e2e33aeb4227820ab0cbe8179e6e0ec05d8eb864d417905c8d08` |
| `v2x__V24__model.pth` | v2x/V24 | 108477603 | `082ee37e61e3cbb90cd8f0d2d54723fc985981e1941e1715852d49dee9ac2ac6` |
| `v2x__V33__model.pth` | v2x/V33 | 108475575 | `b1f312872f8a05f81c7d2dd46549225eb610dfb19e1d84fe02d8e58b56d54ec6` |
| `v2x__V29__model.pth` | v2x/V29 | 108477603 | `289db28255bc2414be78beaa01c5b01d1c6d613d41551c9442790339758b5e49` |
| `v2x__V15__model.pth` | v2x/V15 | 108475575 | `bb3171ef26eecf689b6aac62b5d8c005eb788c77a566781b763f4f2f97dd0284` |
| `v2x__VR1__model.pth` | v2x/VR1 | 240273065 | `2ec938c0efa7ab9a3909c435fa9e4f991d6066a402d4a7b71cd8c046f2767107` |
| `v2x__B_cp16_dens__model.pth` | v2x/B_cp16_dens | 97860765 | `f9cc54eec44afb26c5eaceaef0c58045e6ec0a73b022bc33995c8c3d32daa98c` |
| `v2x__V2__model.pth` | v2x/V2 | 97860764 | `b4c423d058f8578a56da7c16ae8a890d0646bc065980713f2b37d4ef7fba7f1e` |
| `v2x__V36__model.pth` | v2x/V36 | 108475575 | `efe26ec2d10dd7e7b96823d44f6fb4ee8f964a1ae3af30ab601209e4e4caafee` |
| `v2x__NEW_V__model.pth` | v2x/NEW_V | 108477604 | `1065fadfc344cb4f661bb11eb5abbce4943181b3309d1785d46403efbac88642` |
| `v2x__R24_truck__model.pth` | v2x/R24_truck | 108477604 | `c1d256e78415883705664be88d6efb6e889163c428eb095507ec58340f3e1dbb` |
| `v2x__R28_truck__model.pth` | v2x/R28_truck | 108475576 | `6cd31dda750cbcc149525072ddc08cce22aa0aea972e16c05fc9babe9c8940ab` |
| `v2x__PR1_R14__model.pth` | v2x/PR1_R14 | 108475576 | `d9586969e16230deddff8fc20491170fe160de72d3cadddf313b9f577eaa6a5b` |
| `lumpi_1052645_predictions.json` | lumpi/1052645 | 13861535 | `975d977e057edf24e590ea4d7d138a71d78b7858ca974a158d7f911817291f61` |
| `submission_lumpi_WINc3_win1m.zip` | lumpi/1052645 | 30412156 | `fce59963ab206f92d1f6783e5f53343e8a723ce2156ecb0f9879148e17cef217` |
| `v2x_1054488_predictions.json` | v2x/1054488 | 7898536 | `1db3c468a3b05189cf15b14dee95fc2812d5d203e7760157b6c95b9dadf423e4` |
| `submission_v2x_G77_H3.zip` | v2x/1054488 | 29451597 | `87bbc644868f57e77ac65a2a91ef393ca984e06cecf23680c92c5a59eb378214` |
| `v2x__V3__model.pth` | v2x/V3 | 97858739 | `71cf56a336cd35ab8a43ac20ddfb98240be7d6cc959ddb2568ab5e81597b2e2a` |
