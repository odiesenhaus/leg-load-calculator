def test_pretty_names():
    from leg.names import pretty
    assert pretty('sit_to_stand_static') == 'Sit to Stand Static'
    assert pretty('knee_act') == 'Knee Actuator' and pretty('ankle_rod_240mm') == 'Ankle Rod 240mm'
