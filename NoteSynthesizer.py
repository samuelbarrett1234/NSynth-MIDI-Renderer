from typing import Tuple, List, Optional
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from pretty_midi import PrettyMIDI
from librosa import load as load_wav
from librosa.core import resample
from scipy.io.wavfile import write as write_wav


MIDI_MIN_PITCH = 22
MIDI_MAX_PITCH = 108
NSYNTH_DEFAULT_SAMPLE_RATE = 16000
NSYNTH_VELOCITIES = [25, 50, 100, 127]


class NoteSynthesizer:
    def __init__(self,
                 dataset_path: Path,
                 *,
                 sr: int = 44100,
                 transpose: int=0,
                 leg_stac: float=.9,
                 velocities = np.arange(0, 128),
                 preset: int = 0):

        if not dataset_path.is_dir():
            raise ValueError(f"Cannot find directory `{dataset_path}`")

        self.dataset_path = dataset_path
        self.sr = sr
        self.transpose = transpose
        self.leg_stac = leg_stac
        self.velocities = np.array(velocities, dtype=np.int32)
        self.preset = preset
        self.preloaded = False

    def render_sequence(self,
                        midi_filename: Path,
                        *,
                        instrument: str = 'guitar',
                        source_type: str = 'acoustic',
                        preset: Optional[int] = None,
                        playback_speed: float = 1.0,
                        duration_scale: float = 1.0,
                        transpose: Optional[int] = None,
                        eps: float = 1.0e-9):

        if preset is None:
            preset = self.preset

        if transpose is None:
            transpose = self.transpose

        seq, end_time = self._read_midi(midi_filename)
        total_length = int(end_time * self.sr / playback_speed)

        data = np.zeros(total_length)
        for note, velocity, note_start, note_end in seq:
            start_sample = int(note_start * total_length)
            end_sample = int(note_end * total_length)
            duration = end_sample - start_sample

            if duration_scale != 1:
                duration = int(duration * duration_scale)
                end_sample = start_sample + duration

            note_filename = self._get_note_name(note=note+transpose, 
                                                velocity=velocity, 
                                                instrument=instrument, 
                                                source_type=source_type,
                                                preset=preset)

            if not self.preloaded:
                note_filename = self.dataset_path / note_filename

            note = self._render_note(note_filename, duration, velocity)

            if end_sample <= len(data) and duration == len(note):
                data[start_sample:end_sample] += note

            elif duration > len(note) and end_sample <= len(data):
                data[start_sample:start_sample+len(note)] += note

        data /= np.max(np.abs(data)) + eps

        return data, self.sr 

    def preload_notes(self,
                      instrument: str,
                      source_type: str,
                      preset: Optional[int] = None):

        if self.preloaded:
            return  # already preloaded

        if preset is None:
            preset = self.preset

        self.notes = {}
        for n in range(MIDI_MIN_PITCH, MIDI_MAX_PITCH):
            for v in self.velocities:
                note_name = self._get_note_name(n, v, instrument, source_type, preset)
                try:
                    audio, _ = load_wav(self.dataset_path / note_name, sr=self.sr)
                except:
                    audio = None
                self.notes[note_name] = audio

        self.preloaded = True

    def _get_note_name(self,
                       note: int,
                       velocity: int,
                       instrument: str,
                       source_type: str,
                       preset: Optional[int] = None) -> str:

        if preset is None:
            preset = self.preset

        return f"{instrument}_{source_type}_{preset:3d}_{note:3d}_{velocity:3d}"

    def _quantize(self, velocity: int) -> int:

        return self.velocities[np.argmin(np.abs(self.velocities - velocity))]

    def _read_midi(self, filename: Path) -> Tuple[List[Tuple[int, int, float, float]], float]:

        if not filename.is_file():
            raise ValueError(f"Cannot find file `{filename}`")

        midi_data = PrettyMIDI(filename)
        end_time = midi_data.get_end_time()
        
        sequence = []
        for instrument in midi_data.instruments:
            for note in instrument.notes:
                if note.start < end_time:
                    note.velocity = self._quantize(note.velocity)
                    sequence.append((
                        note.pitch,
                        note.velocity,
                        note.start / end_time,
                        note.end / end_time
                    ))

        return sequence, end_time

    def _render_note(self, note_filename: Path, duration, velocity) -> np.ndarray:

        try:
            if self.preloaded:
                note = self.notes[note_filename]
            else:
                note, _ = load_wav(note_filename)

            decay_ind = int(self.leg_stac * duration)
            envelope = np.exp(-np.arange(len(note) - decay_ind) / 3000.0)
            note[decay_ind:] = np.multiply(note[decay_ind:], envelope)

        except:
            raise ValueError(f"Note note found: `{note_filename}`")

        return note[:duration]


if __name__ == "__main__":
    ap = ArgumentParser()
    ap.add_argument('db', required=True, type=Path,
                    help="Path to the NSynth audios folder. (ex: /NSynth/nsynth-train/audios)")
    ap.add_argument('seq', required=True, type=Path,
                    help="MIDI file (.mid) to be rendered")
    ap.add_argument('output', required=True, type=Path,
                    help="Output filename")
    ap.add_argument('--sr', required=False, default=NSYNTH_DEFAULT_SAMPLE_RATE, type=int,
                    help=("Sample rate of the output (default: 16000, typical for "
                          "professional audio: 44100, 48000)"))
    ap.add_argument('--instrument', required=False, default="guitar", type=str,
                    help="Name of the NSynth instrument. (default: 'guitar')")
    ap.add_argument('--source-type', required=False, default="acoustic", type=str,
                    help="Source type of the NSynth instrument (default: 'acoustic')")
    ap.add_argument('--preset', required=False, default=0, type=int,
                    help="Preset of the NSynth instrument (default: 0)")
    ap.add_argument('--transpose', required=False, default=0, type=int,
                    help="Transpose the MIDI sequence by a number of semitones")
    ap.add_argument('--playback-speed', required=False, default=1, type=float,
                    help="Multiply the sequence length by a scalar (default: 1")
    ap.add_argument('--duration-scale', required=False, default=1, type=float,
                    help="Multiply the note durations by a scalar. (default: 1)")
    ap.add_argument('--preload', required=False, default=True, type=bool,
                    help=("Load all notes in memory before rendering for better performance"
                          " (at least 1 GB of RAM is required)"))
    args = ap.parse_args()

    if not args.db.is_dir():
        print(f"Cannot find directory `{args.db}`")
        exit(1)

    if not args.seq.is_file():
        print(f"Cannot find input MIDI file `{args.seq}`")
        exit(1)

    if not args.output.parent.is_dir():
        print(f"Cannot find directory to write output to: `{args.output.parent}`")
        exit(1)

    synth = NoteSynthesizer(
        args.db,
        sr=args.sr,
        velocities=NSYNTH_VELOCITIES,
        preset=args.preset,
        transpose=args.transpose
    )

    if args.preload:
        synth.preload_notes(
            args.instrument, args.source_type
        )

    y, _ = synth.render_sequence(
        sequence=args.seq,
        instrument=args.instrument,
        source_type=args.source_type,
        playback_speed=args.playback_speed,
        duration_scale=args.duration_scale
    )

    if args.sr != NSYNTH_DEFAULT_SAMPLE_RATE:
        y = resample(y, NSYNTH_DEFAULT_SAMPLE_RATE, args.sr)

    write_wav(args.output, args.sr, np.array(32000.0 * y, np.short))
