import React from 'react'
import Downshift from 'downshift'
import { throttle, debounce } from "throttle-debounce"

const nace = require('./item2tekst.json')

export default class Search extends React.Component {
    constructor(props) {
        super(props);
        this.state = { 
            items: [],
            nace: []
        };
        this.autocompleteSearchDebounced = debounce(500, this.autocompleteSearch);
        this.autocompleteSearchThrottled = throttle(500, this.autocompleteSearch);
    }

    _autocompleteCache = {};

    componentDidMount() {
     //   import('./item2tekst.json').then(
     //     res => this.setState({ nace: res })
     //   );
    }

    onChange = (selection) => {
        console.log(`You selected ${selection.value}`)
    }

    onInputValueChange = (inputValue) => {
      console.log(`Input value ${inputValue}`)
      if (inputValue.length < 5 || inputValue.endsWith(' ')) {
        this.autocompleteSearchThrottled(inputValue);
      } else { 
        this.autocompleteSearchDebounced(inputValue);
      }
    }

    autocompleteSearch = (q) => {
        if (typeof q === 'string' || q instanceof String) {
            const url = window.location.href + 'api?q=' + q.toLowerCase()
            //console.log(`url: ${url}`)
            const cached = this._autocompleteCache[url];
            if (cached) {
            return Promise.resolve(cached).then(results => {
                    this.setState({items: results});
                });
            }

            this.waitingFor = q;
            fetch(url)
                .then(response => {
                if (response.status === 200) {
                    if (q === this.waitingFor) {
                        console.log('Content type', response.headers.get("content-type"))
                        const contentType = response.headers.get("content-type");
                        if (contentType && contentType.indexOf("application/json") !== -1) {
                          response.json()
                            .then(results => {
                                this.setState({items: results});
                          });
                        } else {
                          return response.text().then(text => {
                            console.log('Search response', text)
                          });
                        }
                    }
                }
            })
        } else {
            this.setState({items: [{nace: 'Vennligst skriv inn tekst i søkefeltet'}]})
        }
    }

    itemToString = (item) => {
        // const {nace} = this.state;
        // hente tekst for kode fra map i item2tekst.json
        console.log('item to string: ', item)
        const entry = nace.filter(function(o) {
            return o.nace === item.nace;
        });

        if (entry.length === 0 || !('tekst' in entry[0]) ) {
            return item.nace
        }
   
        return item.nace + ': ' + entry[0].tekst;
  
    }

    render() {

        const {items} = this.state;

        return (
            <div>
                <Downshift
                    onChange={selection => this.onChange(selection)}
                    onInputValueChange={(inputValue) => this.onInputValueChange(inputValue)}
                    itemToString={item => (item ? item.pred + ' ' + item.value : '')}
                    >
                    {({
                        getInputProps,
                        getItemProps,
                        getLabelProps,
                        getMenuProps,
                        isOpen,
                        inputValue,
                        highlightedIndex,
                        selectedItem,
                    }) => (
                       <div>
                        <label {...getLabelProps()}></label>
                        <textarea {...getInputProps(
                        {
                            style: {
                            fontSize: '0.9em', 
                            height: 160,
                            verticalAlign: 'middle',
                            width: '80%',
                            },
                        })} 
                        placeholder='Skriv eller lim inn en beskrivelse av aktiviteten for å få forslag til næringskode'
                        />
                        <ul {...getMenuProps()}>
                            {isOpen
                            ? items
                                .map((item, index) => (
                                    <li
                                    {...getItemProps({
                                        key: index,
                                        index,
                                        item,
                                        style: {
                                        listStyleType: 'none',
                                        fontSize: '1.2em',   
                                        backgroundColor:
                                            highlightedIndex === index ? 'darkgray' : 'black',
                                        fontWeight: selectedItem === item ? 'bold' : 'normal',
                                        },
                                    })}
                                    >
                                    {this.itemToString(item)}
                                    </li>
                                ))
                            : null}
                        </ul>
                        </div>
                    )}
                    </Downshift>
                </div>
        )}
}

