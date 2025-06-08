from mesa.visualization.modules import TextElement

class LegendElement(TextElement):
    def render(self, model):
        return (
            "<b>Legenda:</b><br>"
            "<span style='color:red;'>■</span> TumorCell<br>"
            "<span style='color:blue;'>■</span> TCell<br>"
            "<span style='color:cyan;'>■</span> NKCell<br>"
            "<span style='color:green;'>■</span> Macrophage<br>"
            "<span style='color:yellow;'>■</span> THelper<br>"
            "<span style='color:lightblue;'>■</span> Other ImmuneCell"
        )