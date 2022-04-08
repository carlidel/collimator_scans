### Files:

- `colimator-scans-details.pkl`
- `colimator-scans-data.pkl`

both are python pickles, use:

```python
import pickle
dataFromFile = pickle.load(open('colimator-scans-details.pkl','rb'))
```

data inside is dict with the structures defined below:

### Scan details Files

Part of the scan details that are saved in the details file:
```
6052:
    {'beam1':
    {'vertical': {'startTime': '2017-08-06 22:40:00',
    'endTime': '2017-08-07 00:50:00',
    'stepsRange': [4, 51],
    'measuredEmittance': 4.0,
    'nominalBeamSizeUM': 200,
    'dataFileName': 'local_data/6052_MD_Scraping_BLM_TCP_D6_L7_B1.csv',
    'movingJaw': 'TCP.D6L7.B1:MEAS_LVDT_LU',
    'nonMovingJaw': 'TCP.D6L7.B1:MEAS_LVDT_RU'},

   'horizontal':
    {'startTime': '2017-08-06 22:40:00',
    'endTime': '2017-08-07 00:50:00',
    'stepsRange': [4, 40],
    'measuredEmittance': 3.5,
    'nominalBeamSizeUM': 280,
    'dataFileName': 'local_data/6052_MD_Scraping_BLM_TCP_C6_L7_B1.csv',
    'movingJaw': 'TCP.C6L7.B1:MEAS_LVDT_LU',
    'nonMovingJaw': 'TCP.C6L7.B1:MEAS_LVDT_RU'}},
    (...)
```


### Content of the data file

-  6052
	 - TCP_IR7_B1V
		 - lowres
			 - timestamps
			 - TCP.D6L7.B1:MEAS_LVDT_LU
			 - LHC.BCTDC.A6R4.B1:BEAM_INTENSITY
			 - LHC.BCTDC.A6R4.B2:BEAM_INTENSITY
			 - TCP.D6R7.B2:MEAS_LVDT_LU
		 - hires
			 - BLMTI.06L7.B1E10_TCP.D6L7.B1:LOSS_FAST
			 - LHC.BCTDC.A6R4.B1:BEAM_INTENSITY
			 - LHC.BCTDC.A6R4.B2:BEAM_INTENSITY
			 - BLMTI.06L7.B1E10_TCP.C6L7.B1:LOSS_RS09
			 - BLMEI.06L7.B1E10_TCP.A6L7.B1:LOSS_RS09
		 - steps
			 - TCP.D6L7.B1:MEAS_LVDT_LD
			 - TCP.D6L7.B1:MEAS_LVDT_LU
			 - TCP.D6L7.B1:MEAS_LVDT_RD
			 - TCP.D6L7.B1:MEAS_LVDT_RU
	 - TCP_IR7_B1H
		 - lowres
			 - timestamps
			 - TCP.C6L7.B1:MEAS_LVDT_LU
			 - LHC.BCTDC.A6R4.B1:BEAM_INTENSITY
			 - LHC.BCTDC.A6R4.B2:BEAM_INTENSITY
			 - TCP.C6R7.B2:MEAS_LVDT_LU
		 - hires
			 - BLMTI.06L7.B1E10_TCP.C6L7.B1:LOSS_FAST
			 - LHC.BCTDC.A6R4.B1:BEAM_INTENSITY
			 - LHC.BCTDC.A6R4.B2:BEAM_INTENSITY
			 - BLMEI.06L7.B1E10_TCP.A6L7.B1:LOSS_RS09
			 - BLMTI.06L7.B1E10_TCP.D6L7.B1:LOSS_RS09
		 - steps
			 - TCP.C6L7.B1:MEAS_LVDT_LD
			 - TCP.C6L7.B1:MEAS_LVDT_LU
			 - TCP.C6L7.B1:MEAS_LVDT_RD
			 - TCP.C6L7.B1:MEAS_LVDT_RU
-  6194
	 - TCP_IR7_B1V
		 - lowres
			 - timestamps
			 - TCP.D6L7.B1:MEAS_LVDT_LU
			 - LHC.BCTDC.A6R4.B1:BEAM_INTENSITY
			 - LHC.BCTDC.A6R4.B2:BEAM_INTENSITY
			 - TCP.D6R7.B2:MEAS_LVDT_LU
		 - hires
			 - BLMTI.06L7.B1E10_TCP.D6L7.B1:LOSS_FAST
			 - LHC.BCTDC.A6R4.B1:BEAM_INTENSITY
			 - LHC.BCTDC.A6R4.B2:BEAM_INTENSITY
			 - BLMTI.06L7.B1E10_TCP.C6L7.B1:LOSS_RS09
			 - BLMEI.06L7.B1E10_TCP.A6L7.B1:LOSS_RS09
		 - steps
			 - TCP.D6L7.B1:MEAS_LVDT_LD
			 - TCP.D6L7.B1:MEAS_LVDT_LU
			 - TCP.D6L7.B1:MEAS_LVDT_RD
			 - TCP.D6L7.B1:MEAS_LVDT_RU
	 - TCP_IR7_B1H
		 - lowres
			 - timestamps
			 - TCP.C6L7.B1:MEAS_LVDT_LU
			 - LHC.BCTDC.A6R4.B1:BEAM_INTENSITY
			 - LHC.BCTDC.A6R4.B2:BEAM_INTENSITY
			 - TCP.C6R7.B2:MEAS_LVDT_LU
		 - hires
			 - BLMTI.06L7.B1E10_TCP.C6L7.B1:LOSS_FAST
			 - LHC.BCTDC.A6R4.B1:BEAM_INTENSITY
			 - LHC.BCTDC.A6R4.B2:BEAM_INTENSITY
			 - BLMEI.06L7.B1E10_TCP.A6L7.B1:LOSS_RS09
			 - BLMTI.06L7.B1E10_TCP.D6L7.B1:LOSS_RS09
		 - steps
			 - TCP.C6L7.B1:MEAS_LVDT_LD
			 - TCP.C6L7.B1:MEAS_LVDT_LU
			 - TCP.C6L7.B1:MEAS_LVDT_RD
			 - TCP.C6L7.B1:MEAS_LVDT_RU
-  7221
	 - TCP_IR7_B1V
		 - lowres
			 - timestamps
			 - TCP.D6L7.B1:MEAS_LVDT_LU
			 - LHC.BCTDC.A6R4.B1:BEAM_INTENSITY
			 - LHC.BCTDC.A6R4.B2:BEAM_INTENSITY
			 - TCP.D6R7.B2:MEAS_LVDT_LU
		 - hires
			 - BLMTI.06L7.B1E10_TCP.D6L7.B1:LOSS_RS09
			 - LHC.BCTDC.A6R4.B1:BEAM_INTENSITY
			 - LHC.BCTDC.A6R4.B2:BEAM_INTENSITY
			 - BLMTI.06L7.B1E10_TCP.C6L7.B1:LOSS_RS09
			 - BLMEI.06L7.B1E10_TCP.A6L7.B1:LOSS_RS09
		 - steps
			 - TCP.D6L7.B1:MEAS_LVDT_LD
			 - TCP.D6L7.B1:MEAS_LVDT_LU
			 - TCP.D6L7.B1:MEAS_LVDT_RD
			 - TCP.D6L7.B1:MEAS_LVDT_RU
-  7392
	 - TCP_IR7_B1V
		 - lowres
			 - timestamps
			 - TCP.D6L7.B1:MEAS_LVDT_LU
			 - LHC.BCTDC.A6R4.B1:BEAM_INTENSITY
			 - LHC.BCTDC.A6R4.B2:BEAM_INTENSITY
			 - TCP.D6R7.B2:MEAS_LVDT_LU
		 - hires
			 - BLMTI.06L7.B1E10_TCP.D6L7.B1:LOSS_RS09
			 - LHC.BCTDC.A6R4.B1:BEAM_INTENSITY
			 - LHC.BCTDC.A6R4.B2:BEAM_INTENSITY
			 - BLMTI.06L7.B1E10_TCP.C6L7.B1:LOSS_RS09
			 - BLMEI.06L7.B1E10_TCP.A6L7.B1:LOSS_RS09
		 - steps
			 - TCP.D6L7.B1:MEAS_LVDT_LD
			 - TCP.D6L7.B1:MEAS_LVDT_LU
			 - TCP.D6L7.B1:MEAS_LVDT_RD
			 - TCP.D6L7.B1:MEAS_LVDT_RU
