from itertools import chain


################################################
################   NEGATION TRIGGERS  ###################
################################################

ContextD_triggers = {'afwezigheid van',
                     'evenmin',
                     'gedaald',
                     'geen',
                     'geen aanwijzingen voor',
                     'geen klachten van',
                     'geen oorzaak van',
                     'geen teken van',
                     'geen tekenen van',
                     'heeft geen',
                     'kan niet',
                     'leek niet',
                     'niet',
                     'niet als',
                     'onbekend',
                     'uitsluiten',
                     'verdwenen',
                     'vertonen geen',
                     'vertoonde geen',
                     'vrij van',
                     'weg',
                     'zonder',
                     'zonder tekenen van'}

negation_pre_triggers = {'core_triggers': {'geen', 'niet', 'zonder', 'noch'},
                         'added_triggers': {'nooit', 'géén', 'nihil'}}
all_negation_pre_triggers = set(chain.from_iterable(negation_pre_triggers.values()))

negation_post_triggers = {'core_triggers': {'geen', 'niet', 'nee', 'neen'},
                          'added_triggers': {'nooit', 'géén', 'nihil', 'negatief'}}
all_negation_post_triggers = set(chain.from_iterable(negation_post_triggers.values()))

negation_triggers = {'pre': all_negation_pre_triggers,
                     'post': all_negation_post_triggers,
                     'all': all_negation_pre_triggers.union(all_negation_post_triggers)}
