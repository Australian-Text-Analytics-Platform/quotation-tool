from abc import ABCMeta, abstractmethod


class Viz(metaclass=ABCMeta):
    @abstractmethod
    def render(self):
        """ Renders the visualisation. """
        raise NotImplementedError()


class Widget(metaclass=ABCMeta):
    @abstractmethod
    def widget(self):
        """ Display the interactive widget. """
        raise NotImplementedError()
