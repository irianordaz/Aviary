"""External subsystem builder that creates an engine deck from a CSV data file."""

from aviary.subsystems.propulsion.engine_deck import EngineDeck
from aviary.utils.aviary_values import AviaryValues
from aviary.utils.functions import get_path
from aviary.variable_info.variables import Aircraft


class EngineTableBuilder(EngineDeck):
    """
    EngineDeck builder that reads engine performance data from a CSV file.

    Parameters
    ----------
    name : str
        Label for this engine model.
    csv_path : str
        Path to engine performance CSV. Accepts paths relative to the Aviary models
        directory (e.g. 'models/engines/turbofan_22k.csv') or absolute paths.
    options : AviaryValues, optional
        Additional engine options forwarded to EngineDeck. All required options not
        provided here are filled from Aviary metadata defaults by _preprocess_inputs.
    """

    def __init__(
        self,
        name: str = 'engine_table',
        csv_path: str = 'models/engines/turbofan_22k.csv',
        options: AviaryValues = None,
    ):
        if options is None:
            options = AviaryValues()
        options.set_val(Aircraft.Engine.DATA_FILE, get_path(csv_path))
        super().__init__(name=name, options=options)
