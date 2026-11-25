# eForecast project author: George Sideratos
# License for eForecast contributions: MIT
# Header date: 2026-11-25

Layer_name_list = ['conv',
                   'conv_3d',
                   'time_distr_conv',
                   'time_distr_conv_3d',
                   'lstm',
                   'hidden_dense',
                   'dense',
                   'Flatten',
                   'Dropout']
experiments = dict()

experiments['exp_fuzzy1'] = {'row_all': [('dense', {1})],
                             'output': [
                                 ('concatenate', []),
                                 ('dense', {1})
                             ],
                             }
experiments['cnn1'] = {
                        'nwp': [('cnn_att', 2),
                                ('conv', 2),
                                 ('Flatten', []),
                                ('dense', 512),
                                ('dense', 128),
                                 ],
                        'row_calendar': [
                            ('dense', 64),
                        ],
                        'output': [('cross_attention', {1}),
                                   ('dense', {64})
                                   ]
                        }
experiments['transformer1'] =  {'nwp': [('cnn_att', 'cbam'),
                                 ('cnn_att', 'att'),
                                 ('cnn_att', 'geodatt'),
                                 ('conv', 2),
                                 ('Flatten', []),
                                 ('dense', 512),
                                 ('dense', 128),
                                 ],
                         'row_calendar': [
                             ('dense', {64}),
                         ],
                         'output': [
                             ('cross_attention', {1}),
                             ('dense', {64})
                         ]}

experiments['transformer2'] =  {'nwp': [('cnn_att', 'cbam'),
                                 ('cnn_att', 'att'),
                                 ('cnn_att', 'geodatt'),
                                 ('conv', 2),
                                 ('cnn_att', 'cbam'),
                                 ('cnn_att', 'geodatt'),
                                 ('Flatten', []),
                                 ('dense', 512),
                                 ('dense', 128),
                                 ],
                         'row_calendar': [
                             ('dense', {64}),
                         ],
                         'output': [
                             ('cross_attention', {1}),
                             ('dense', {64})
                         ]}

experiments['transformer3'] =  {'nwp': [('cnn_att', 'cbam'),
                                 ('cnn_att', 'att'),
                                 ('conv', 2),
                                 ('Flatten', []),
                                 ('dense', 512),
                                 ('dense', 128),
                                 ],
                         'row_calendar': [
                             ('dense', {64}),
                         ],
                         'output': [
                             ('cross_attention', {1}),
                             ('dense', {64})
                         ]}

experiments['mlp1'] = {

    'nwp': [('conv', 2),
            ('conv', 2),
            ('Flatten', []),
            ('dense', 1024),
            ('dense', 128),
            ],
    'row_calendar': [
                     ('dense', {64}),
                     ],
    'output': [('cross_attention', {1}),
               ('dense', {64})
               ],
}

experiments['mlp2'] = {

            'nwp': [
                    ('cnnT', 2),
                    ('cnnT', 2),
                    ('Flatten', []),
                    ('dense', 512),
                    ('dense', 128),
                    ],
            'row_calendar': [
                ('dense', {64}),
            ],
            'output': [('cross_attention', {1}),
                       ('dense', {64})
                       ],
        }

experiments['mlp3'] = {

            'nwp': [
                    ('cnnT', 2),
                    ('conv', 2),
                    ('Flatten', []),
                    ('dense', 512),
                    ('dense', 128),
                    ],
            'row_calendar': [
                ('dense', {64}),
            ],
            'output': [('cross_attention', {1}),
                       ('dense', {64})
                       ],
        }

experiments['mlp4'] =  {

            'nwp': [
                    ('cnnT', 2),
                    ('Flatten', []),
                    ('dense', 1024),
                    ('dense', 128),
                    ],
            'row_calendar': [
                ('dense', {64}),
            ],
            'output': [('cross_attention', {1}),
                       ('dense', {64})
                       ],
        }

# experiments['timm_net'] = {
#     'nwp': [('vit_net', 1),
#                ('Flatten', []),
#                ('dense', 2048),
#                ('dense', 720)
#                ],
#     'row_calendar': [('dense', 4),
#                      ],
#     'output': [('concatenate', []),
#                ('dense', 64),
#                ]
# }

experiments['lstm1'] = {'nwp': [('cnn_att', 2),
                                 ('bi_lstm', 8),
                                    ('Flatten', []),
                                    ('dense', {512}),
                                    ('dense', {128}),
                                     ],
                        'row_calendar': [
                            ('dense', {120}),
                        ],
                            'output': [
                                ('cross_attention', {1}),
                                ('dense', {128}),
                            ],
                        }

# experiments['lstm2'] = {'lstm': [('lstm_att', 8),
#                                      ('Flatten', []),
#                                     ('dense', {512}),
#                                      ],
#                             'output': [
#                                 ('concatenate', []),
#                                 ('dense', {128}),
#                                 ('dense', {32})
#                             ],
#                         }
# experiments['lstm3'] = { 'lstm': [('bi_lstm', 48),
#                          ('Flatten', []),
#                          ('dense', 512),
#                          ],
#                         'output': [
#                             ('dense', 128),
#                             ('dense', {32})
#                         ]}
# experiments['mlp5'] = {
#     'lstm': [('bi_lstm', [5, 8]),
#              ('Flatten', []),
#              ('dense', {512}),
#              ],
#     'output': [
#         ('concatenate', []),
#         ('dense', {128})
#     ],
# }
#
# experiments['mlp6'] = {
#     'lstm': [('bi_lstm', [2, 4]),
#              ('Flatten', []),
#             ('dense', {256}),
#              ],
#     'output': [
#         ('cross_attention', {1}),
#         ('dense', {32})
#     ],
# }

# experiments['distributed_cnn1'] = {
#     'nwp': [('conv', [3, 10]),
#             ('Flatten', []),
#             ('dense', {0.25, 0.125}),
#             ('dense', {0.25, 0.125}),
#             ],
#     # 'row_all': [('dense', {5, 0.5, 1, 2}),
#     #             ('dense', {64, }),
#     #             ],
#     'row_calendar': [('dense', {5, 10, 20}),
#                      ('dense', {64}),
#                      ],
#     'output': [('concatenate', []),
#                ('dense', {0.5, 0.25}),
#                ('dense', {0.5, 'linear'}),
#                ('dense', {32, 64, 128})
#                ],
# }

# experiments['distributed_mlp1'] = {
#     'row_nwp': [('dense', {4, 8}),
#                 ('dense', {0.25, 0.5}),
#                 ],
#     # 'row_all': [('dense', {5, 0.5, 1, 2}),
#     #             ('dense', {64, }),
#     #             ],
#     'row_calendar': [('dense', {5, 10, 20}),
#                      ('dense', {64}),
#                      ],
#     'output': [('concatenate', []),
#                ('dense', {0.5, 0.25}),
#                ('dense', {0.5, 'linear'}),
#                ('dense', {32, 64, 128})
#                ],
# }
# experiments['distributed_mlp2'] = {
#     'row_nwp': [('dense', {4, 8}),
#                 ('dense', {0.25, 0.5}),
#                 ],
#     # 'row_all': [('dense', {5, 0.5, 1, 2}),
#     #             ('dense', {64, }),
#     #             ],
#     'row_calendar': [('dense', {5, 10, 20}),
#                      ('dense', {64}),
#                      ],
#     'output': [('concatenate', []),
#                ('dense', {0.5, 0.25}),
#                ('dense', {0.5, 'linear'}),
#                ('dense', {32, 64, 128})
#                ],
# }
# experiments['mlp_for_combine_data'] = {
#     'row_all': [('dense', {4, 8}),
#                 ('dense', {0.5, 0.75}),
#                 ],
#     'output': [('concatenate', []),
#                ('dense', {0.5, 0.75}),
#                ('dense', {32, 64, 128})
#                ],
# }
# experiments['output_combine'] = {
#     'output': [('concatenate', []),
#                ('dense', {32})
#                ]
# }
# mlp_for_combine_simple = {
#     'row_all': [('dense', {4, 8}),
#                 ],
#     'output': [('concatenate', []),
#                ('dense', {0.5, 0.75}),
#                ('dense', {32, 64, 128})
#                ],
# }
# experiments['distributed_mlp3'] = {
#     'row_all': [('dense', {0.25, 0.125}),
#                 ],
#     'output': [('concatenate', []),
#                ('dense', {0.5, 0.25}),
#                ('dense', {0.5, 'linear'}),
#                ('dense', {32, 64, 128})
#                ],
# }

# experiments['lstm1'] = {
#     'lstm': [('lstm', [2, 10]),
#              ('lstm', [2, 5]),
#              ('Flatten', []),
#              ('dense', [3, 7])],
#     'output': [('concatenate', []),
#                ('dense', {64, 128})]
# }
# experiments['lstm2'] = {
#     'lstm': [('lstm', [2, 10]),
#              ('Flatten', []),
#              ('dense', [3, 7]),
#              ('Reshape', []),  #: [32, 32]}
#              ('lstm', [2, 10]),
#              ('Flatten', []),
#              ('dense', [3, 7])
#              ],
#     'output': [('concatenate', []),
#                ('dense', {64, 128})
#                ]
# }
# experiments['lstm3'] = {
#     'lstm': [('lstm', [2, 10]),
#              ('Flatten', []),
#              ('dense', [3, 7]),
#
#              ],
#     'output': [('concatenate', []),
#                ('dense', {12, 64, 128}),
#                ]
# }
# experiments['lstm4'] = {
#     'lstm': [('lstm', {0.5, 1, 2}),
#              ('Flatten', []),
#              ('dense', {1, 0.5}),
#              ('dense', {1, 0.5, 0.25})],
#     'output': [('concatenate', []),
#                ('dense', {12, 64, 128})
#                ]
# }
# experiments['transformer'] = {
#     'lstm': [('transformer', 256),
#              ('Flatten', [])],
#     # ('dense', {1, 0.5}),
#     # ('dense', {1, 0.5, 0.25})],
#     'output': [('concatenate', []),
#                ('dense', 256),
#                ('dense', 64)
#                ]
# }
# experiments['timm_net'] = {
#     'images': [('vit_net', 1),
#                ('Flatten', []),
#                ('dense', 2048),
#                ('dense', 720),
#                ],
#     'row_calendar': [('dense', 4),
#                          ],
#     'output': [('concatenate', []),
#                ('dense', 64),
#                ]
# }
# experiments['timm_net1a'] = {
#     'images': [('vit_net', 1),
#                ('Flatten', []),
#                ('dense', 1024),
#                ('dense', 256),
#                ],
#     'row_calendar': [('dense', 28),
#                      ],
#     'output': [('cross_attention', {1}),
#                ('dense', 64),
#                ]
# }
# experiments['timm_net1b'] = {
#     'images': [('vit_net', 1),
#                ('Flatten', []),
#                ('dense', 2048),
#                ('dense', 720),
#                ],
#     'row_calendar': [('dense', 4),
#                      ],
#     'output': [('concatenate', []),
#                ('dense', 64),
#                ]
# }
# experiments['timm_net2a'] = {
#     'images': [('vit_net', 1),
#                ('Flatten', []),
#                ('dense', 1024),
#                ('dense', 256)
#                ],
#     'row_obs_calendar': [('dense', 4),
#                  ('dense', 64)
#                  ],
#     'output': [('cross_attention', {1}),
#                ('dense', 64),
#                ]
# }
# experiments['timm_net2b'] = {
#     'images': [('vit_net', 1),
#                ('Flatten', []),
#                ('dense', 1024),
#                ('dense', 256)
#                ],
#     'row_obs_calendar': [('dense', 4),
#                  ('dense', 64)
#                  ],
#     'output': [('concatenate', []),
#                ('dense', 64),
#                ]
# }
# experiments['timm_net3a'] = {
#     'images': [('time_distr_vit_net', 1),
#                ('Flatten', []),
#                ('dense', 2048),
#                ('dense', 720)
#                ],
#     'row_calendar': [('dense', 4),
#                      ],
#     'output': [('concatenate', []),
#                ('dense', 64),
#                ]
# }

# experiments['CrossViVit_net'] = {
#     'images': [('vit_net', 1),
#                ('Flatten', []),
#                ('dense', 1024),
#                ('dense', 128)
#                ],
#
#     'nwp': [('conv', {0, 1, 2, 3, 4, 5}),
#             ('conv', {0, 1, 3}),
#             ('Flatten', []),
#             ('dense', {32, 64, 128}),
#             ],
#     'row_calendar': [
#         ('dense', {32, 64, 128}),
#     ],
#     'row_all': [
#         ('dense', {32, 64, 128}),
#     ],
#     'output': [('concatenate', 1),
#                ('dense', 64),
#                ]
# }
# experiments['Time_CrossViVit_net'] = {
#     'images': [('time_distr_vit_net', 1),
#                ('Flatten', []),
#                ('dense', 1024),
#                ('dense', 128)
#                ],
#     'row_calendar': [('dense', 64),
#                      ],
#     'output': [
#         ('concatenate', []),
#         ('dense', 64),
#     ]
# }
# experiments['trans_net'] = {
#     'lstm': [('transformer', 128),
#              ('Flatten', []),
#              ('dense', 720),
#              ('dense', 128)
#              ],
#     'output': [
#         ('concatenate', []),
#         ('dense', 64),
#     ]
# }
#
# experiments['yolo'] = {
#     'images': [('yolo', 1),
#                ('Flatten', []),
#                ('dense', 1024),
#                ('dense', 256)
#                ],
#     'row_calendar': [('dense', 64),
#                      ],
#     # 'row_obs_nwp': [('dense', 4),
#     #              ('dense', 720),
#     #              ('dense', 128)
#     #              ],
#     'output': [('concatenate', []), ('dense', 64),
#                ]
# }
# experiments['unet'] = {
#     'images': [('unet', 1),
#                ('Flatten', []),
#                ('dense', 1024),
#                ('dense', 64)
#                ],
#     'row_calendar': [('dense', 64),
#                      ],
#     # 'row_obs_nwp': [('dense', 4),
#     #              ('dense', 720),
#     #              ('dense', 128)
#     #              ],
#     'output': [('concatenate', []), ('dense', 256),
#                ]
# }
#
# experiments['distributed_lstm1'] = {
#     'lstm': [('lstm', 1),
#              ('Flatten', []),
#              ('dense', 0.25),
#              ('dense', 0.5),
#              ],
#     'output': [('concatenate', []), ('dense', 0.25),
#                ('dense', 0.5),
#                ('dense', 32)
#                ],
# }

# 2026-09-09: attention over the NWP grid (one token per cell) merged with the calendar by cross attention
experiments['grid1'] = {'nwp': [('grid_transformer', 64),
                                ('Flatten', []),
                                ('dense', 128),
                                ],
                        'row_calendar': [
                            ('dense', 64),
                        ],
                        'output': [('cross_attention', 1),
                                   ('dense', 64)
                                   ]
                        }
# encoder-decoder LSTM with attention over the past steps (lstm data), merged with the calendar
experiments['lstm_att1'] = {'lstm': [('lstm_att', 2),
                                     ('Flatten', []),
                                     ('dense', 128),
                                     ],
                            'row_calendar': [
                                ('dense', 64),
                            ],
                            'output': [('cross_attention', 1),
                                       ('dense', 64)
                                       ]
                            }

# TimeXer encoder on the lstm data (endogenous patches, future covariates as exogenous tokens) merged
# with the calendar by cross attention: the tokens of the encoder are the keys, the calendar the query.
experiments['timexer1'] = {'lstm': [('timexer', {32, 64}),
                                    ('Flatten', []),
                                    ('dense', 128),
                                    ],
                           'row_calendar': [
                               ('dense', 64),
                           ],
                           'output': [('cross_attention', 1),
                                      ('dense', 64)
                                      ]
                           }

# TiDE on the lstm data (dense encoder, future covariates per step) with the calendar, concatenated:
# MLP only, cheap, the candidate cluster model.
experiments['tide1'] = {'lstm': [('tide', {16, 32}),
                                 ('Flatten', []),
                                 ('dense', 64),
                                 ],
                        'row_calendar': [
                            ('dense', 32),
                        ],
                        'output': [('concatenate', []),
                                   ('dense', 64)
                                   ]
                        }

# AFNO on the NWP grid (global Fourier mixing of the cells), one token per cell into a cross-attention
# merge queried by the calendar: the spectral alternative to cnn1 and grid1.
experiments['afno1'] = {'nwp': [('afno', {32, 64}),
                                ('tokens', []),
                                ('dense', 64),
                                ],
                        'row_calendar': [
                            ('dense', 64),
                        ],
                        'output': [('cross_attention', 1),
                                   ('dense', 64)
                                   ]
                        }

# Perceiver merge with one query per horizon step over the NWP cells (AFNO tokens), the past steps
# (TimeXer tokens) and the calendar: every lead time reads its own mix of the modalities.
experiments['perceiver1'] = {'nwp': [('afno', 32),
                                     ('tokens', []),
                                     ],
                             'lstm': [('timexer', 32),
                                      ],
                             'row_calendar': [
                                 ('dense', 32),
                             ],
                             'output': [('perceiver', {32, 64}),
                                        ('dense', 64)
                                        ]
                             }

# TSMixer on the lstm data (time and feature mixing MLPs, future covariates per step) with the calendar,
# concatenated: the second cheap cluster model, next to tide1.
experiments['tsmixer1'] = {'lstm': [('tsmixer', {16, 32}),
                                    ('Flatten', []),
                                    ('dense', 64),
                                    ],
                           'row_calendar': [
                               ('dense', 32),
                           ],
                           'output': [('concatenate', []),
                                      ('dense', 64)
                                      ]
                           }
