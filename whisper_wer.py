import os
import jiwer
import whisper
from docx import Document
from typing import List, Optional
import re


class Record:
    def __init__(self, audio_record_path: str, transcript_record_path: str):
        self._audio_path = audio_record_path
        self._transcript_path = transcript_record_path

    @property
    def audio_path(self) -> str:
        return self._audio_path

    @property
    def transcript_path(self) -> str:
        return self._transcript_path

    @property
    def record_name(self) -> str:
        # read the file name without extension as the record name
        return self.audio_path.split("/")[-1].split(".")[0]

    def get_transcript_content(self) -> str:
        doc = Document(self.transcript_path)
        full_text = [p.text for p in doc.paragraphs]
        return "\n".join(full_text)

    def get_audio_transcription(self, whisper_model) -> str:
        result = whisper_model.transcribe(self.audio_path)
        return result["text"]

    def cal_wer(self, whisper_model):
        audio_transcription = self.get_audio_transcription(whisper_model)
        reference_transcript = self.get_transcript_content()
        wer = jiwer.wer(reference_transcript, audio_transcription)

        return wer

    def wer_output(self, whisper_model):
        return f"{self.record_name} - Word Error Rate (WER): {self.cal_wer(whisper_model) * 100:.2f}%"


class Participant:
    """Correspond to folder like Nut01"""

    def __init__(self, audio_folder: str, transcript_folder: str):
        self._audio_foler = audio_folder
        self._transcript_foler = transcript_folder
        self.records = []

    def get_transcript_files(self):
        return sorted(
            [f for f in os.listdir(self._transcript_foler) if not f.startswith(".")]
        )

    def get_audio_files(self, exclude_pattern: str = "_ASA24_"):
        return sorted(
            [
                f
                for f in os.listdir(self._audio_foler)
                if exclude_pattern not in f and not f.startswith(".")
            ]
        )

    @property
    def name(self):
        return self._transcript_foler.split("/")[-1].split(" ")[0]

    def get_records(self) -> List[Record]:
        if self.records:
            return self.records

        transcripts = self.get_transcript_files()
        audios = self.get_audio_files()
        common = sorted(
            list(
                set(t.split(".")[0] for t in transcripts)
                & set(a.split(".")[0] for a in audios)
            )
        )

        # we expect audio and transcript files share the same name
        # Note: we only care about the participant that has both audio and transcript data
        self.records = [
            Record(
                audio_record_path=os.path.join(self._audio_foler, f + ".wav"),
                transcript_record_path=os.path.join(
                    self._transcript_foler, f + ".docx"
                ),
            )
            for f in common
        ]

        return self.records


class Session:
    def __init__(self, audio_folder: str, transcript_folder: str):
        self._audio_folder = audio_folder
        self._transcript_folder = transcript_folder
        self.participants: Participant = []

    @property
    def name(self):
        return self._audio_folder.split("/")[-1]

    def get_participant_audio_folders(self):
        return sorted(
            [d for d in os.listdir(self._audio_folder) if not d.startswith(".")]
        )

    def get_participant_transcript_folders(self):
        return sorted(
            [d for d in os.listdir(self._transcript_folder) if not d.startswith(".")]
        )

    def get_participants(self) -> List[Participant]:
        if self.participants:
            return self.participants
        participant_audios = self.get_participant_audio_folders()
        participant_transcripts = self.get_participant_transcript_folders()

        # we expect audio and transcript subfolders share the name name
        # Note: we only care about the participant that has both audio and transcript data
        self.participants = [
            Participant(
                audio_folder=os.path.join(self._audio_folder, d),
                transcript_folder=os.path.join(self._transcript_folder, d),
            )
            for d in list(set(participant_audios) & set(participant_transcripts))
        ]

        return self.participants


class DataProcessor:
    def __init__(self, audio_folder: str, transcript_folder: str, model):
        self._audio_folder = audio_folder
        self._transcript_folder = transcript_folder
        self._model = model

    def get_sessions(self) -> List[Session]:
        session_names = sorted(
            sn
            for sn in list(
                set(os.listdir(self._audio_folder))
                & set(os.listdir(self._transcript_folder))
            )
            if not sn.startswith(".")
        )
        return [
            Session(
                audio_folder=os.path.join(self._audio_folder, sn),
                transcript_folder=os.path.join(self._transcript_folder, sn),
            )
            for sn in session_names
        ]

    def process(
        self, output_file: Optional[str] = None, write_to_file: Optional[bool] = True
    ):
        if not output_file:
            output_file = os.path.join(
                os.path.dirname(self._audio_folder), "output.txt"
            )

        res = []
        for session in self.get_sessions():
            for participant in session.get_participants():
                for record in participant.get_records():
                    res.append(record.wer_output(self._model))

        res.sort()
        if write_to_file:
            with open(output_file, "w") as f:
                for r in res:
                    f.write(f"{r}\n")

        return res


if __name__ == "__main__":
    data_root_path = "projects/nana_food_records/data"
    model = whisper.load_model("base")
    processor = DataProcessor(
        audio_folder=data_root_path + "/audios",
        transcript_folder=data_root_path + "/transcripts",
        model=model,
    )
    res = processor.process()
    for r in res:
        print(r)
