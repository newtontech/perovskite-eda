"""molecular_plots.py

Molecular structure visualization using RDKit.
Placeholder for advanced molecular plotting; will be expanded with
2D depictions, substructure highlighting, and property annotations.
"""

import warnings
from pathlib import Path
from typing import Any

warnings.filterwarnings("ignore")


class MolecularPlotter:
    """Generate molecular structure figures."""

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._has_rdkit = False
        try:
            from rdkit import Chem
            from rdkit.Chem import Draw
            self._has_rdkit = True
        except ImportError:
            pass

    def depict_molecules(self, smiles_list: list[str], legends: list[str] | None = None,
                         title: str = "Molecular Structures", filename: str = "molecules.png") -> Path | None:
        """Generate a grid of 2D molecular depictions."""
        if not self._has_rdkit or not smiles_list:
            return None
        from rdkit import Chem
        from rdkit.Chem import Draw

        mols = []
        valid_legends = []
        for i, smi in enumerate(smiles_list):
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                mols.append(mol)
                valid_legends.append(legends[i] if legends and i < len(legends) else f"Mol {i+1}")
        if not mols:
            return None

        img = Draw.MolsToGridImage(mols, molsPerRow=min(4, len(mols)), legends=valid_legends,
                                   subImgSize=(250, 250), returnPNG=False)
        path = self.output_dir / filename
        img.save(path)
        return path

    def highlight_substructure(self, smiles: str, smarts: str,
                               filename: str = "highlight.png") -> Path | None:
        """Highlight a substructure pattern in a single molecule."""
        if not self._has_rdkit:
            return None
        from rdkit import Chem
        from rdkit.Chem import Draw

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        patt = Chem.MolFromSmarts(smarts)
        if patt is None:
            return None
        match = mol.GetSubstructMatch(patt)
        if not match:
            return None

        d = Draw.rdMolDraw2D.MolDraw2DCairo(400, 400)
        d.drawOptions().highlightRadius = 0.3
        Draw.rdMolDraw2D.PrepareAndDrawMolecule(d, mol, highlightAtoms=match)
        d.FinishDrawing()
        path = self.output_dir / filename
        with open(path, "wb") as f:
            f.write(d.GetDrawingText())
        return path
