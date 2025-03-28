import React from 'react';

class DataChoiceComponent extends React.Component<{ onChoiceMade: any }, { text: string }> {
  constructor(props: any) {
    super(props);
    this.state = { text: '' };
    this.handleChange = this.handleChange.bind(this);
    this.handleChoice = this.handleChoice.bind(this);
  }

  render() {
    return (
      <div className="flex items-center gap-2">
        <label htmlFor="data-choice">
          Which dataset do you want to use? (moons, blobs, circles)
        </label>
        <input
          id="data-choice"
          className="rounded-md border-2 border-gray-300 bg-white p-2"
          onChange={this.handleChange}
          value={this.state.text}
        />
        <button onClick={this.handleChoice}>Confirm</button>
      </div>
    );
  }

  handleChange(e: any) {
    this.setState({ text: e.target.value });
  }

  handleChoice(e: any) {
    console.log('Previous state: ', this.state);
    this.props.onChoiceMade(this.state.text);
    this.setState({
      text: '',
    });
  }
}
export default DataChoiceComponent;
