# CMI-Detect-Befavio-With-Sensor-Data
腕につけたウェアラブルデバイスのセンサーデータから、その人がどんな動きをしているかを分類するコンペティション
## ファイル構成
src/models.py　# モデルアーキテクチャ用のスクリプト
src/preprocess.py # 前処理用のスクリプト
src/train.py 訓練用のスクリプト
src/utils.py
src/config.py Hydraのデフォルト値はconfig,py記述するものとする
その他 src内に適宜追加してもよい

# 訓練アプローチ
* データセットは、8151個の時系列データで構成される。sequence_idがシーケンスを決める
* fold分けはgloupkfoldでsubject毎
* 特徴量エンジニアリングを行ってから、モデルに入れる。データリークには気をつけて。scaler等も保存する必要があるかも？？
* ハイパーパラメータはHydraを使って参照
* Pytorchを使ったニューラルネットワーク
* 評価は(二値分類 + 多クラス分類 ) / 2になりますが、訓練は多クラス分類だけにフォーカスすればいいです。
* Optimizerは、Adam, AdamW, RAdamScheduleFreeからhydraで選べるようにして、デフォルトはRAdamScheduleFree
    * RAdamScheduleFreeについては詳しくはこっちを調べて　https://github.com/facebookresearch/schedule_free
* Schedulerはwarmup + cosineアニーリングがデフォルトで
* timmのEMA V3を利用すること 0.99がデフォルトで

## 前処理
* 前処理クラスPreprocessorを作成する。
* 訓練データにfit_transformを行ったあと、検証データにtransformを行うことで前処理を行う
* 正規化はstanderd scaler

## 基本原則
* 実験管理はhydra, wandb, loggerを持ち入る
* wandbはエポックごとに、lr, loss, 他クラス分類スコア, 初期家事
* 出力ファイルはcfg.env.exp_output_dirへ
* 実行は docker-compose run --rm kaggle python experiments/expXXX/run.py
    * これがだめなら、uv run python experiments/expXXX/run.py

## inputファイル
cfg.env.input_dir / 
train.csv
train_demographics.csv
test.csv
test_demographics.csv

## コンペの説明
Description
Body-focused repetitive behaviors (BFRBs), such as hair pulling, skin picking, and nail biting, are self-directed habits involving repetitive actions that, when frequent or intense, can cause physical harm and psychosocial challenges. These behaviors are commonly seen in anxiety disorders and obsessive-compulsive disorder (OCD), thus representing key indicators of mental health challenges.

hand

To investigate BFRBs, the Child Mind Institute has developed a wrist-worn device, Helios, designed to detect these behaviors. While many commercially available devices contain Inertial Measurement Units (IMUs) to measure rotation and motion, the Helios watch integrates additional sensors, including 5 thermopiles (for detecting body heat) and 5 time-of-flight sensors (for detecting proximity). See the figure to the right for the placement of these sensors on the Helios device.

We conducted a research study to test the added value of these additional sensors for detecting BFRB-like movements. In the study, participants performed series of repeated gestures while wearing the Helios device:

They began a transition from “rest” position and moved their hand to the appropriate location (Transition);

They followed this with a short pause wherein they did nothing (Pause); and

Finally they performed a gesture from either the BFRB-like or non-BFRB-like category of movements (Gesture; see Table below).

Each participant performed 18 unique gestures (8 BFRB-like gestures and 10 non-BFRB-like gestures) in at least 1 of 4 different body-positions (sitting, sitting leaning forward with their non-dominant arm resting on their leg, lying on their back, and lying on their side). These gestures are detailed in the table below, along with a video of the gesture.

BFRB-Like Gesture (Target Gesture)	Video Example
Above ear - Pull hair	Sitting
Forehead - Pull hairline	Sitting leaning forward
Forehead - Scratch	Sitting
Eyebrow - Pull hair	Sitting
Eyelash - Pull hair	Sitting
Neck - Pinch skin	Sitting
Neck - Scratch	Sitting
Cheek - Pinch skin	Sitting,
Sitting leaning forward,
Lying on back,
Lying on side


Non-BFRB-Like Gesture (Non-Target Gesture)	Video Example
Drink from bottle/cup	Sitting
Glasses on/off	Sitting
Pull air toward your face	Sitting
Pinch knee/leg skin	Sitting leaning forward
Scratch knee/leg skin	Sitting leaning forward
Write name on leg	Sitting leaning forward
Text on phone	Sitting
Feel around in tray and pull out an object	Sitting
Write name in air	Sitting
Wave hello	Sitting

This competition challenges you to develop a predictive model capable of distinguishing (1) BFRB-like gestures from non-BFRB-like gestures and (2) the specific type of BFRB-like gesture. Critically, when your model is evaluated, half of the test set will include only data from the IMU, while the other half will include all of the sensors on the Helios device (IMU, thermopiles, and time-of-flight sensors).

Your solutions will have direct real-world impact, as the insights gained will inform design decisions about sensor selection — specifically whether the added expense and complexity of thermopile and time-of-flight sensors is justified by significant improvements in BFRB detection accuracy compared to an IMU alone. By helping us determine the added value of these thermopiles and time-of-flight sensors, your work will guide the development of better tools for detection and treatment of BFRBs.

Relevant articles: Garey, J. (2025). What Is Excoriation, or Skin-Picking? Child Mind Institute. https://childmind.org/article/excoriation-or-skin-picking/

Martinelli, K. (2025). What is Trichotillomania? Child Mind Institute. https://childmind.org/article/what-is-trichotillomania/

Evaluation
The evaluation metric for this contest is a version of macro F1 that equally weights two components:

Binary F1 on whether the gesture is one of the target or non-target types.
Macro F1 on gesture, where all non-target sequences are collapsed into a single non_target class
The final score is the average of the binary F1 and the macro F1 scores.

If your submission includes a gesture value not found in the train set your submission will trigger an error.

Submission File
You must submit to this competition using the provided evaluation API, which ensures that models perform inference on a single sequence at a time. For each sequence_id in the test set, you must predict the corresponding gesture.

## データセットの説明
Dataset Description
In this competition you will use sensor data to classify body-focused repetitive behaviors (BFRBs) and other gestures.

This dataset contains sensor recordings taken while participants performed 8 BFRB-like gestures and 10 non-BFRB-like gestures while wearing the Helios device on the wrist of their dominant arm. The Helios device contains three sensor types:

1x Inertial Measurement Unit (IMU; BNO080/BNO085): An integrated sensor that combines accelerometer, gyroscope, and magnetometer measurements with onboard processing to provide orientation and motion data.
5x Thermopile Sensor (MLX90632): A non-contact temperature sensor that measures infrared radiation.
5x Time-of-Flight Sensor (VL53L7CX): A sensor that measures distance by detecting how long it takes for emitted infrared light to bounce back from objects.
You must submit to this competition using the provided Python evaluation API, which serves test set data one sequence at a time. To use the API, follow the example in this notebook.

Expect approximately 3,500 sequences in the hidden test set. Half of the hidden-test sequences are recorded with IMU only; the thermopile (thm_) and time-of-flight (tof__v*) columns are still present but contain null values for those sequences. This will allow us to determine whether adding the time-of-flight and thermopile sensors improves our ability to detect BFRBs. Note also that there is known sensor communication failure in this dataset, resulting in missing data from some sensors in some sequences.

Files
[train/test].csv

row_id
sequence_id - An ID for the batch of sensor data. Each sequence includes one Transition, one Pause, and one Gesture.
sequence_type - If the gesture is a target or non-target type. Train only.
sequence_counter - A counter of the row within each sequence.
subject - A unique ID for the subject who provided the data.
gesture - The target column. Description of sequence Gesture. Train only.
orientation - Description of the subject's orientation during the sequence. Train only.
behavior - A description of the subject's behavior during the current phase of the sequence.
acc_[x/y/z] - Measure linear acceleration along three axes in meters per second squared from the IMU sensor.
rot_[w/x/y/z] - Orientation data which combines information from the IMU's gyroscope, accelerometer, and magnetometer to describe the device's orientation in 3D space.
thm_[1-5] - There are five thermopile sensors on the watch which record temperature in degrees Celsius. Note that the index/number for each corresponds to the index in the photo on the Overview tab.
tof_[1-5]_v[0-63] - There are five time-of-flight sensors on the watch that measure distance. In the dataset, the 0th pixel for the first time-of-flight sensor can be found with column name tof_1_v0, whereas the final pixel in the grid can be found under column tof_1_v63. This data is collected row-wise, where the first pixel could be considered in the top-left of the grid, with the second to its right, ultimately wrapping so the final value is in the bottom right (see image above). The particular time-of-flight sensor is denoted by the number at the start of the column name (e.g., 1_v0 is the first pixel for the first time-of-flight sensor while 5_v0 is the first pixel for the fifth time-of-flight sensor). If there is no sensor response (e.g., if there is no nearby object causing a signal reflection), a -1 is present in this field. Units are uncalibrated sensor values in the range 0-254. Each sensor contains 64 pixels arranged in an 8x8 grid, visualized in the figure below.


[train/test]_demographics.csv

These tabular files contain demographic and physical characteristics of the participants.

subject
adult_child: Indicates whether the participant is a child (0) or an adult (1). Adults are defined as individuals aged 18 years or older.
age: Participant's age in years at time of data collection.
sex: Participants sex assigned at birth, 0= female, 1 = male.
handedness: Dominant hand used by the participant, 0 = left-handed, 1 = right-handed.
height_cm: Height of the participant in centimeters.
shoulder_to_wrist_cm: Distance from shoulder to wrist in centimeters.
elbow_to_wrist_cm: Distance from elbow to wrist in centimeters.
sample_submission.csv

sequence_id
gesture
kaggle_evaluation/ The files that implement the evaluation API. You can run the API locally on a Unix machine for testing purposes.

## 本コンペの評価指標
```python
"""
Hierarchical macro F1 metric for the CMI 2025 Challenge.

This script defines a single entry point `score(solution, submission, row_id_column_name)`
that the Kaggle metrics orchestrator will call.
It performs validation on submission IDs and computes a combined binary & multiclass F1 score.
"""

import pandas as pd
from sklearn.metrics import f1_score


class ParticipantVisibleError(Exception):
    """Errors raised here will be shown directly to the competitor."""
    pass


class CompetitionMetric:
    """Hierarchical macro F1 for the CMI 2025 challenge."""
    def __init__(self):
        self.target_gestures = [
            'Above ear - pull hair',
            'Cheek - pinch skin',
            'Eyebrow - pull hair',
            'Eyelash - pull hair',
            'Forehead - pull hairline',
            'Forehead - scratch',
            'Neck - pinch skin',
            'Neck - scratch',
        ]
        self.non_target_gestures = [
            'Write name on leg',
            'Wave hello',
            'Glasses on/off',
            'Text on phone',
            'Write name in air',
            'Feel around in tray and pull out an object',
            'Scratch knee/leg skin',
            'Pull air toward your face',
            'Drink from bottle/cup',
            'Pinch knee/leg skin'
        ]
        self.all_classes = self.target_gestures + self.non_target_gestures

    def calculate_hierarchical_f1(
        self,
        sol: pd.DataFrame,
        sub: pd.DataFrame
    ) -> float:

        # Validate gestures
        invalid_types = {i for i in sub['gesture'].unique() if i not in self.all_classes}
        if invalid_types:
            raise ParticipantVisibleError(
                f"Invalid gesture values in submission: {invalid_types}"
            )

        # Compute binary F1 (Target vs Non-Target)
        y_true_bin = sol['gesture'].isin(self.target_gestures).values
        y_pred_bin = sub['gesture'].isin(self.target_gestures).values
        f1_binary = f1_score(
            y_true_bin,
            y_pred_bin,
            pos_label=True,
            zero_division=0,
            average='binary'
        )

        # Build multi-class labels for gestures
        y_true_mc = sol['gesture'].apply(lambda x: x if x in self.target_gestures else 'non_target')
        y_pred_mc = sub['gesture'].apply(lambda x: x if x in self.target_gestures else 'non_target')

        # Compute macro F1 over all gesture classes
        f1_macro = f1_score(
            y_true_mc,
            y_pred_mc,
            average='macro',
            zero_division=0
        )

        return 0.5 * f1_binary + 0.5 * f1_macro


def score(
    solution: pd.DataFrame,
    submission: pd.DataFrame,
    row_id_column_name: str
) -> float:
    """
    Compute hierarchical macro F1 for the CMI 2025 challenge.

    Expected input:
      - solution and submission as pandas.DataFrame
      - Column 'sequence_id': unique identifier for each sequence
      - 'gesture': one of the eight target gestures or "Non-Target"

    This metric averages:
    1. Binary F1 on SequenceType (Target vs Non-Target)
    2. Macro F1 on gesture (mapping non-targets to "Non-Target")

    Raises ParticipantVisibleError for invalid submissions,
    including invalid SequenceType or gesture values.


    Examples
    --------
    >>> import pandas as pd
    >>> row_id_column_name = "id"
    >>> solution = pd.DataFrame({'id': range(4), 'gesture': ['Eyebrow - pull hair']*4})
    >>> submission = pd.DataFrame({'id': range(4), 'gesture': ['Forehead - pull hairline']*4})
    >>> score(solution, submission, row_id_column_name=row_id_column_name)
    0.5
    >>> submission = pd.DataFrame({'id': range(4), 'gesture': ['Text on phone']*4})
    >>> score(solution, submission, row_id_column_name=row_id_column_name)
    0.0
    >>> score(solution, solution, row_id_column_name=row_id_column_name)
    1.0
    """
    # Validate required columns
    for col in (row_id_column_name, 'gesture'):
        if col not in solution.columns:
            raise ParticipantVisibleError(f"Solution file missing required column: '{col}'")
        if col not in submission.columns:
            raise ParticipantVisibleError(f"Submission file missing required column: '{col}'")

    metric = CompetitionMetric()
    return metric.calculate_hierarchical_f1(solution, submission)
```